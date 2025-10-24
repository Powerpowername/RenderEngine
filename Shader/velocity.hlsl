#include "constantBuffers.hlsl"
struct VertexIn
{
    float3 Position         : POSITION;
    float3 Normal           : NORMAL;
    float3 Tangent          : TANGENT;
    float3 Bitangent        : BITANGENT;
    float2 TexCoord         : TEXCOORD0;
};
/*
💡 为何要分开定义？
关键在于 SV_POSITION这个系统值语义的特殊性。
在像素着色器中，使用 SV_POSITION语义修饰的输入变量得到的是当前像素在屏幕上的​​整数坐标​​（更准确地说，是经过一个偏移的插值坐标），
它​​不再精确对应​​你在顶点着色器中输出的那个原始裁剪空间坐标。而运动向量的计算需要的就是顶点着色器输出的、未经光栅化特殊处理的​​原始裁剪空间坐标​​。
*/
struct VertexOut
{
    float4 PositionH        : SV_POSITION;      // Clip space position.
    float4 CurrPositionH    : POSITION0;        // Current clip space position. Using SV_POSITION directly results in strange result.
    float4 PrevPositionH    : POSITION1;        // Previous clip space position.
};

VertexOut VS(VertexIn vin)
{
    VertexOut vout;
    
    float4 positionW = mul(float4(vin.Position, 1.0), g_World);
    vout.PositionH = mul(positionW, g_ViewProj);
    
    vout.CurrPositionH = vout.PositionH;

    float4 prevPositionW = mul(float4(vin.Position, 1.0), g_PrevWorld);
    vout.PrevPositionH = mul(prevPositionW, g_PrevViewProj);
    
    return vout;

}


float2 PS(VertexOut pin) : SV_Target
{
    float3 currPosNDC = pin.CurrPositionH.xyz / pin.CurrPositionH.w;
    float3 prevPosNDC = pin.PrevPositionH.xyz / pin.PrevPositionH.w;

    //g_Jitter​​亚像素级别的抖动，此处看不出先前哪里加上的抖动，其实抖动是加在g_PrevViewProj上的（知乎文章中是加在这里的，实际需要根据源码确认）
    float2 velocity = (currPosNDC.xy - g_Jitter) - (prevPosNDC.xy - g_PreviousJitter);
    return velocity;

}


