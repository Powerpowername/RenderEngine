#include "samplers.hlsl"

static const float FLT_EPS = 0.00000001f;
static const float PI = 3.141592653589793;


struct Resources
{
    uint SourceTexIndex;
    uint HistoryTexIndex;
    uint DepthTexIndex;
    uint VelocityTexIndex;
};
//5.1版本过后的HLSL支持ConstantBuffer
ConstantBuffer<Resources> g_Resources : register(b6);

struct VertexOut
{
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

VertexOut VS(uint vertexID : SV_VertexID)
{
    //绘制一个覆盖整个屏幕的三角形，因为光栅化的前提必须要有顶点数据输入，否则不会进行光栅化
    VertexOut vout;
    
    // draw a triangle that covers the entire screen
    const float2 tex = float2(uint2(vertexID, vertexID << 1) & 2);
    vout.position = float4(lerp(float2(-1, 1), float2(1, -1), tex), 0, 1);
    vout.texcoord = tex;
    
    return vout;
}

//滤波器------​生成一个平滑的权重值，该值基于输入的距离 x，距离越近权重越高
float FilterBlackmanHarris(in float x)
{
    x = 1.0f - x;

    const float a0 = 0.35875f;
    const float a1 = 0.48829f;
    const float a2 = 0.14128f;
    const float a3 = 0.01168f;
    return saturate(a0 - a1 * cos(PI * x) + a2 * cos(2 * PI * x) - a3 * cos(3 * PI * x));
}
//实现​​高质量纹理过滤
float4 SampleTextureCatmullRom(in Texture2D<float4> tex, in SamplerState linearSampler, in float2 uv, in float2 texSize)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float4 result = 0.0f;
    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos0.y), 0.0f) * w0.x * w0.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos0.y), 0.0f) * w12.x * w0.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos0.y), 0.0f) * w3.x * w0.y;

    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos12.y), 0.0f) * w0.x * w12.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos12.y), 0.0f) * w12.x * w12.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos12.y), 0.0f) * w3.x * w12.y;

    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos3.y), 0.0f) * w0.x * w3.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos3.y), 0.0f) * w12.x * w3.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos3.y), 0.0f) * w3.x * w3.y;

    return result;
}

float Luminance(float3 color)
{
    return dot(color, float3(0.2127, 0.7152, 0.0722));
}


float4 PS(VertexOut pin) : SV_Target
{
    //载入纹理资源
    Texture2D source = ResourceDescriptorHeap[g_Resources.SourceTexIndex];
    Texture2D history = ResourceDescriptorHeap[g_Resources.HistoryTexIndex];
    Texture2D depth = ResourceDescriptorHeap[g_Resources.DepthTexIndex];
    Texture2D velocity = ResourceDescriptorHeap[g_Resources.VelocityTexIndex];
    int width, height, level;
    source.GetDimensions(0, width, height, level);

    //初始化相关状态    
    float3 sourceSampleTotal = float3(0, 0, 0);
    float sourceSampleWeight = 0.0;
    
    float3 neighborhoodMin = 99999999.0;
    float3 neighborhoodMax = -99999999.0;
    
    float3 m1 = float3(0, 0, 0);
    float3 m2 = float3(0, 0, 0);
    
    float closestDepth = 99999999.0;
    int2 closestDepthPixelPosition = int2(0, 0);

    //3x3邻域采样循环：收集当前像素周围信息[1](@ref)
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int2 pixelPosition = pin.texcoord * float2(width, height) + int2(x, y);//将坐标转换为像素坐标
            pixelPosition = clamp(pixelPosition, 0, int2(width, height) - 1);  // 钳位坐标防止越界
            float3 neighbor = max(0, source[pixelPosition].rgb);  // 采样邻域颜色，确保非负
            float subSampleDistance = length(float2(x, y));  // 计算到中心像素的距离
            float subSampleWeight = FilterBlackmanHarris(subSampleDistance);  // 根据距离获取权重（距离越近权重越高）
            // 累加加权颜色和权重
            sourceSampleTotal += neighbor * subSampleWeight;
            sourceSampleWeight += subSampleWeight;
            // 更新邻域颜色极值
            neighborhoodMin = min(neighborhoodMin, neighbor);
            neighborhoodMax = max(neighborhoodMax, neighbor);
            // 累加统计矩
            m1 += neighbor;
            m2 += neighbor * neighbor;//计算方差使用
            // 更新最近深度及对应像素位置
            float currentDepth = depth[pixelPosition].r;
            if (currentDepth < closestDepth) {
                closestDepth = currentDepth;
                closestDepthPixelPosition = pixelPosition;
            }
        }
    }
    // *0.5的目的是将速度向量映射到[0, 1]，*-0.5是将y轴翻转，因为NDC空间中的坐标是向上的，UV中是向下的
    float2 motionVector = velocity[closestDepthPixelPosition].xy * float2(0.5, -0.5);
    float2 historyTexCoord = pin.texcoord - motionVector;

    float3 sourceSample = sourceSampleTotal / sourceSampleWeight;//获取加权平均颜色
    //检测纹理坐标是否越界，越界就返回当前帧颜色，否则与历史帧颜色进行混合
    if (any(historyTexCoord != saturate(historyTexCoord)))
        return float4(sourceSample, 1.0);

    float3 historySample = SampleTextureCatmullRom(history, g_SamplerLinearWrap, historyTexCoord, float2(width, height)).rgb;
    float invSampleCount = 1.0 / 9.0;
    //采用正态分布的思想，落在1σ的范围外，可以解决闪烁问题，但处理鬼影效果较弱
    float gamma = 1.0;
    float3 mu = m1 * invSampleCount;
    float3 sigma = sqrt(abs((m2 * invSampleCount) - (mu * mu)));
    float3 minc = mu - gamma * sigma;
    float3 maxc = mu + gamma * sigma;
    //此处实际上没有用上minc和maxc
    historySample = clamp(historySample, neighborhoodMin, neighborhoodMax);
    
    float sourceWeight = 0.05;
    float historyWeight = 1.0 - sourceWeight;
    //此处为HDR处理，将颜色压缩到[0, 1]，并计算亮度，然后进行混合
    float3 compressedSource = sourceSample * rcp(max(max(sourceSample.r, sourceSample.g), sourceSample.b) + 1.0);
    float3 compressedHistory = historySample * rcp(max(max(historySample.r, historySample.g), historySample.b) + 1.0);
    float luminanceSource = Luminance(compressedSource);
    float luminanceHistory = Luminance(compressedHistory);
    //亮度权重，防止除0，越亮区域权重越小
    /*
    权重调整过程如下：
​    ​调整当前帧权重​​：
    高亮度导致 luminanceSource很大。
    sourceWeight从 0.05被削减到一个更小的值，比如 0.02。
​    ​调整历史帧权重​​：
    高亮度导致 luminanceHistory很大。
    historyWeight从 0.95被削减到一个较小的值，比如 0.20。
    ​​最终混合效果​​：
    ​调整前混合​​：结果 ≈ 5% 的正确当前颜色 + 95% 的错误历史鬼影颜色 → ​​鬼影非常明显​​。
​    ​调整后混合​​：结果 ≈ 2% 的正确当前颜色 + 20% 的错误历史鬼影颜色 → 经过归一化后，​​错误历史颜色的占比从95%大幅下降到约 0.2 / (0.02+0.2) ≈ 91%​​。
    ​​!! 核心效果​​：虽然两者的权重都被降低了，但历史权重的​​绝对值降幅（从0.95降到0.2）​​ 远远大于当前帧权重的降幅（从0.05降到0.02）。这使得在最终混合时，​​历史帧颜色的影响力被极大地削弱了​​。
    */
    sourceWeight *= 1.0 / (1.0 + luminanceSource);
    historyWeight *= 1.0 / (1.0 + luminanceHistory);
    //加权混合
    float3 result = (sourceSample * sourceWeight + historySample * historyWeight) / max(sourceWeight + historyWeight, 0.00001);
    return float4(result, 1);
}

