#include "utils.hlsl"

cbuffer cbSsao : register(b0)
{
    float4x4 gProj;           // 投影矩阵
    float4x4 gInvProj;        // 逆投影矩阵（用于从NDC空间回到视图空间）
    float4x4 gProjTex;        // 投影纹理矩阵（用于将3D位置投影到纹理坐标）
    
    float4 gBlurWeights[3];   // 模糊权重（用于后续的SSAO模糊处理）
    float2 gInvRenderTargetSize; // 渲染目标尺寸的倒数（用于像素到纹理坐标的转换）
    
    // SSAO核心参数
    float gOcclusionRadius;   // 遮蔽计算半径（决定采样范围）
    float gOcclusionFadeStart; // 遮蔽淡出起始距离
    float gOcclusionFadeEnd;   // 遮蔽淡出结束距离
    float gSurfaceEpsilon;    // 表面容差值（防止自遮蔽），后续查一下表面容差是什么意思
};


cbuffer cbRootConstants : register(b1)  // 根常量缓冲区，用于存储频繁更新的参数
{
    bool gHorizontalBlur;      // 控制模糊方向的标志，true为水平模糊，false为垂直模糊
};


// 纹理资源声明
Texture2D gNormalMap : register(t0);  // 法线纹理，存储视图空间法线信息
Texture2D gDepthMap : register(t1);   // 深度纹理，存储深度信息


// 采样器状态声明，需要在CPU端输入
SamplerState gsamPointClamp : register(s0);  // 点采样+钳位寻址模式，用于精确采样
SamplerState gsamDepthMap : register(s1);    // 深度纹理专用采样器

// 常量定义
static const uint NumSamples = 64;           // 采样点数量，影响质量和性能
static const float InvNumSamples = 1.0 / float(NumSamples); // 采样点数量的倒数，用于求平均值


// 全屏四边形纹理坐标定义（6个顶点构成2个三角形）
static const float2 gTexCoords[6] =
{
    float2(0.0f, 1.0f),  // 左下角
    float2(0.0f, 0.0f),  // 左上角
    float2(1.0f, 0.0f),  // 右上角
    float2(0.0f, 1.0f),  // 左下角（第二个三角形开始）
    float2(1.0f, 0.0f),  // 右上角
    float2(1.0f, 1.0f)   // 右下角
};

// 顶点着色器输出结构
struct VertexOut
{
    float3 PositionV        : POSITION1;        // 视图空间位置，用于后续计算
    float2 TexCoord         : TEXCOORD0;        // 纹理坐标，用于采样纹理
    float4 PositionH        : SV_POSITION;      // 齐次裁剪空间位置，系统必需
};


// 顶点着色器函数
VertexOut VS(uint vid : SV_VertexID)  // 输入为顶点ID
{
    /*
    像素着色器利用插值得到的视图空间射线方向（pin.PositionV）和从深度纹理采样得到的深度值（pz），
    计算出该像素在视图空间中的真实3D位置。计算公式本质上是按深度比例缩放射线方向向量
    */
    VertexOut vout;  // 声明输出变量

    vout.TexCoord = gTexCoords[vid];  // 根据顶点ID获取对应的纹理坐标

    // 将纹理坐标转换为NDC坐标：x从[0,1]到[-1,1]，y从[0,1]到[1,-1]（反转Y轴）
    vout.PositionH = float4(2.0f * vout.TexCoord.x - 1.0f, 1.0f - 2.0f * vout.TexCoord.y, 0.0f, 1.0f);

    // 使用逆投影矩阵将NDC坐标转换回视图空间
    float4 ph = mul(vout.PositionH, gInvProj);
    vout.PositionV = ph.xyz / ph.w;  // 透视除法，得到正确的视图空间坐标，齐次坐标

    return vout;  // 返回顶点着色器输出

}



// 遮蔽计算函数：根据深度差计算遮蔽值
float OcclusionFunction(float distZ)
{
    float occlusion = 0.0f;  // 初始化遮蔽值为0
    
    // 只有当采样点在当前点前方且超过表面容差时才计算遮蔽（将表面抬一抬）
    if (distZ > gSurfaceEpsilon) 
    {
        float fadeLength = gOcclusionFadeEnd - gOcclusionFadeStart;  // 计算衰减范围长度
        
        // 线性衰减计算：从gOcclusionFadeStart到gOcclusionFadeEnd从1衰减到0
        occlusion = saturate((gOcclusionFadeEnd - distZ) / fadeLength);
    }
    
    return occlusion;  // 返回计算得到的遮蔽值
}

// 深度转换函数：将NDC深度[-1,1]转换回视图空间线性深度
float NdcDepthToViewDepth(float z_ndc)
{
    // 使用投影矩阵参数进行转换：z_ndc = A + B/viewZ
    float viewZ = gProj[3][2] / (z_ndc - gProj[2][2]);
    return viewZ;
}




// #include "utils.hlsl"  // 引入工具函数库，包含采样、坐标转换等辅助函数

// cbuffer cbSsao : register(b0)  // 定义常量缓冲区，使用寄存器b0，存储SSAO参数
// {
//     float4x4 gProj;           // 投影矩阵，用于将视图空间坐标转换到裁剪空间
//     float4x4 gInvProj;        // 逆投影矩阵，用于将裁剪空间坐标转换回视图空间
//     float4x4 gProjTex;        // 投影纹理矩阵，用于将3D位置投影到纹理坐标空间

//     float4 gBlurWeights[3];   // 模糊权重数组，用于后续的模糊处理阶段

//     float2 gInvRenderTargetSize; // 渲染目标尺寸的倒数，用于像素到纹理坐标的转换

//     float gOcclusionRadius;   // 遮蔽半径，控制采样点的范围大小
//     float gOcclusionFadeStart; // 遮蔽效果开始衰减的距离
//     float gOcclusionFadeEnd;   // 遮蔽效果完全消失的距离
//     float gSurfaceEpsilon;     // 表面容差值，防止自遮蔽现象
// };

// cbuffer cbRootConstants : register(b1)  // 根常量缓冲区，用于存储频繁更新的参数
// {
//     bool gHorizontalBlur;      // 控制模糊方向的标志，true为水平模糊，false为垂直模糊
// };
 
// // 纹理资源声明
// Texture2D gNormalMap : register(t0);  // 法线纹理，存储视图空间法线信息
// Texture2D gDepthMap : register(t1);   // 深度纹理，存储深度信息

// // 采样器状态声明
// SamplerState gsamPointClamp : register(s0);  // 点采样+钳位寻址模式，用于精确采样
// SamplerState gsamDepthMap : register(s1);    // 深度纹理专用采样器

// // 常量定义
// static const uint NumSamples = 64;           // 采样点数量，影响质量和性能
// static const float InvNumSamples = 1.0 / float(NumSamples); // 采样点数量的倒数，用于求平均值
 
// // 全屏四边形纹理坐标定义（6个顶点构成2个三角形）
// static const float2 gTexCoords[6] =
// {
//     float2(0.0f, 1.0f),  // 左下角
//     float2(0.0f, 0.0f),  // 左上角
//     float2(1.0f, 0.0f),  // 右上角
//     float2(0.0f, 1.0f),  // 左下角（第二个三角形开始）
//     float2(1.0f, 0.0f),  // 右上角
//     float2(1.0f, 1.0f)   // 右下角
// };

// // 顶点着色器输出结构
// struct VertexOut
// {
//     float3 PositionV        : POSITION1;        // 视图空间位置，用于后续计算
//     float2 TexCoord         : TEXCOORD0;        // 纹理坐标，用于采样纹理
//     float4 PositionH        : SV_POSITION;      // 齐次裁剪空间位置，系统必需
// };

// // 顶点着色器函数
// VertexOut VS(uint vid : SV_VertexID)  // 输入为顶点ID
// {
//     VertexOut vout;  // 声明输出变量

//     vout.TexCoord = gTexCoords[vid];  // 根据顶点ID获取对应的纹理坐标

//     // 将纹理坐标转换为NDC坐标：x从[0,1]到[-1,1]，y从[0,1]到[1,-1]（反转Y轴）
//     vout.PositionH = float4(2.0f * vout.TexCoord.x - 1.0f, 1.0f - 2.0f * vout.TexCoord.y, 0.0f, 1.0f);
 
//     // 使用逆投影矩阵将NDC坐标转换回视图空间
//     float4 ph = mul(vout.PositionH, gInvProj);
//     vout.PositionV = ph.xyz / ph.w;  // 透视除法，得到正确的视图空间坐标

//     return vout;  // 返回顶点着色器输出
// }

// // 遮蔽计算函数：根据深度差计算遮蔽值
// float OcclusionFunction(float distZ)
// {
//     float occlusion = 0.0f;  // 初始化遮蔽值为0
    
//     // 只有当采样点在当前点前方且超过表面容差时才计算遮蔽
//     if (distZ > gSurfaceEpsilon) 
//     {
//         float fadeLength = gOcclusionFadeEnd - gOcclusionFadeStart;  // 计算衰减范围长度
        
//         // 线性衰减计算：从gOcclusionFadeStart到gOcclusionFadeEnd从1衰减到0
//         occlusion = saturate((gOcclusionFadeEnd - distZ) / fadeLength);
//     }
    
//     return occlusion;  // 返回计算得到的遮蔽值
// }

// // 深度转换函数：将NDC深度[-1,1]转换回视图空间线性深度
// float NdcDepthToViewDepth(float z_ndc)
// {
//     // 使用投影矩阵参数进行转换：z_ndc = A + B/viewZ
//     float viewZ = gProj[3][2] / (z_ndc - gProj[2][2]);
//     return viewZ;
// }
 
// // 像素着色器函数
// float4 PS(VertexOut pin) : SV_Target  // 输出到渲染目标
// {
//     // 获取当前像素的法线向量（视图空间）并进行归一化
//     float3 N = normalize(gNormalMap.SampleLevel(gsamPointClamp, pin.TexCoord, 0.0f).xyz);
    
//     // 获取当前像素的深度值并转换到视图空间
//     float pz = gDepthMap.SampleLevel(gsamDepthMap, pin.TexCoord, 0.0f).r;
//     pz = NdcDepthToViewDepth(pz);
    
//     // 计算切空间基向量S和T
//     float3 S, T;
//     ComputeBasisVectors(N, S, T);
    
//     // 重建当前像素在视图空间中的完整3D位置
//     // 使用比例关系：p.z / pin.PositionV.z = 实际深度 / 顶点深度
//     float3 p = (pz / pin.PositionV.z) * pin.PositionV;
    
//     float occlusionSum = 0.0f;  // 初始化遮蔽总和
    
//     // 循环采样NumSamples次
//     for (int i = 0; i < NumSamples; ++i)
//     {
//         // 使用Hammersley序列生成低差异的2D采样点
//         float2 uv = SampleHammersley(i, InvNumSamples);
        
//         // 在半球内采样并转换到视图空间
//         float3 offset = TangentToBasis(SampleHemisphere(uv.x, uv.y), N, S, T);
        
//         // 在当前点p周围生成采样点q，使用gOcclusionRadius控制范围
//         float3 q = p + gOcclusionRadius * offset;
        
//         // 将采样点q投影到屏幕空间
//         float4 projQ = mul(float4(q, 1.0f), gProjTex);
//         projQ /= projQ.w;  // 透视除法

//         // 在深度纹理中查询采样点位置的实际深度值
//         float rz = gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r;
//         rz = NdcDepthToViewDepth(rz);  // 转换到视图空间

//         // 重建实际场景点r在视图空间中的位置
//         // 使用比例关系：r.z / q.z = 实际深度 / 采样点深度
//         float3 r = (rz / q.z) * q;
        
//         // 计算深度差：正值表示r在p的前方
//         float distZ = p.z - r.z;
        
//         // 计算法线点积：衡量r相对于p表面的角度，max确保非负
//         float dp = max(dot(N, normalize(r - p)), 0.0f);

//         // 计算遮蔽贡献：综合考虑角度和距离因素
//         float occlusion = dp * OcclusionFunction(distZ);
//         occlusionSum += occlusion;  // 累加到总和
//     }
    
//     occlusionSum /= NumSamples;  // 求平均值
    
//     float access = 1.0f - occlusionSum;  // 计算可及度（1 - 遮蔽度）
    
//     // 使用幂函数增强对比度，使遮蔽效果更加明显
//     return saturate(pow(access, 6.0f));
// }