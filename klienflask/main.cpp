// =====================================================================================
//  klein_flask.cpp  --  Klein Flask Simulator / neuroevolutionary intelligence test
//  single file, C++11, OpenGL 3.3 core, GLEW + GLFW.  No GLM.  No Qt required.
//
//  Linux   : g++ -std=c++11 -O2 klein_flask.cpp -o kfs -lGLEW -lGL -lglfw
//  macOS   : clang++ -std=c++11 -O2 klein_flask.cpp -o kfs -lGLEW -lglfw \
//            -framework OpenGL -framework Cocoa -framework IOKit -framework CoreFoundation
//  Windows : cl /EHsc /O2 /std:c++14 klein_flask.cpp glew32.lib glfw3.lib opengl32.lib ws2_32.lib winmm.lib
//
//  keys: TAB focus chat · ENTER send · LMB orbit · WHEEL zoom · CLICK inside flask = poke
//        F force face-gen · G force agent-gen · H handshake now · N rename · P spawn peer flask
//        M mute · R reset · 1..9 pick flask · / commands: help name say peer broadcast upload wake
// =====================================================================================
 
 //UNTESTED STILL so use at own risk
#define _CRT_SECURE_NO_WARNINGS
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdarg>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include <random>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <mmsystem.h>
  #pragma comment(lib,"ws2_32.lib")
  #pragma comment(lib,"winmm.lib")
  typedef SOCKET Sock;
  static const Sock BADSOCK = INVALID_SOCKET;
  static void sockInit(){ WSADATA w; WSAStartup(MAKEWORD(2,2),&w); }
  static void sockClose(Sock s){ if(s!=BADSOCK) closesocket(s); }
#else
  #include <unistd.h>
  #include <fcntl.h>
  #include <errno.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/types.h>
  typedef int Sock;
  static const Sock BADSOCK = -1;
  static void sockInit(){}
  static void sockClose(Sock s){ if(s!=BADSOCK) ::close(s); }
#endif

static const float PI = 3.14159265358979323846f;

// ------------------------------------------------------------------ math (no glm) ----
struct V3{
    float x,y,z;
    V3():x(0),y(0),z(0){}
    V3(float a,float b,float c):x(a),y(b),z(c){}
    V3 operator+(const V3&o)const{return V3(x+o.x,y+o.y,z+o.z);}
    V3 operator-(const V3&o)const{return V3(x-o.x,y-o.y,z-o.z);}
    V3 operator-()const{return V3(-x,-y,-z);}
    V3 operator*(float s)const{return V3(x*s,y*s,z*s);}
    V3 operator/(float s)const{return V3(x/s,y/s,z/s);}
    V3&operator+=(const V3&o){x+=o.x;y+=o.y;z+=o.z;return *this;}
    V3&operator-=(const V3&o){x-=o.x;y-=o.y;z-=o.z;return *this;}
    float dot(const V3&o)const{return x*o.x+y*o.y+z*o.z;}
    V3 cross(const V3&o)const{return V3(y*o.z-z*o.y,z*o.x-x*o.z,x*o.y-y*o.x);}
    float len()const{return sqrtf(x*x+y*y+z*z);}
    V3 norm()const{float l=len();return l>1e-9f?V3(x/l,y/l,z/l):V3(0,0,0);}
};
static inline V3 operator*(float s,const V3&v){return v*s;}
static inline V3 vmin(const V3&a,const V3&b){return V3(a.x<b.x?a.x:b.x,a.y<b.y?a.y:b.y,a.z<b.z?a.z:b.z);}
static inline V3 vmax(const V3&a,const V3&b){return V3(a.x>b.x?a.x:b.x,a.y>b.y?a.y:b.y,a.z>b.z?a.z:b.z);}
static inline float clampf(float v,float a,float b){return v<a?a:(v>b?b:v);}
static inline float lerp(float a,float b,float t){return a+(b-a)*t;}
static inline float smoothstep(float a, float b, float x){
    float t=clampf((x-a)/(b-a),0.f,1.f);
    return t*t*(3.f-2.f*t);
}

struct M4{ float m[16];
    static M4 id(){M4 r;memset(r.m,0,sizeof(r.m));r.m[0]=r.m[5]=r.m[10]=r.m[15]=1;return r;}
};
static M4 mul(const M4&a,const M4&b){
    M4 r; for(int c=0;c<4;c++) for(int rw=0;rw<4;rw++){
        float s=0; for(int k=0;k<4;k++) s+=a.m[k*4+rw]*b.m[c*4+k];
        r.m[c*4+rw]=s; }
    return r;
}
static M4 persp(float fovyDeg,float asp,float zn,float zf){
    M4 r=M4::id(); float f=1.0f/tanf(fovyDeg*PI/360.0f);
    r.m[0]=f/asp; r.m[5]=f; r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1; r.m[14]=2*zf*zn/(zn-zf); r.m[15]=0;
    return r;
}
static M4 lookAt(V3 e,V3 c,V3 u){
    V3 f=(c-e).norm(), s=f.cross(u).norm(), t=s.cross(f);
    M4 r=M4::id();
    r.m[0]=s.x;r.m[4]=s.y;r.m[8]=s.z;   r.m[12]=-s.dot(e);
    r.m[1]=t.x;r.m[5]=t.y;r.m[9]=t.z;   r.m[13]=-t.dot(e);
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;r.m[14]=f.dot(e);
    return r;
}
static M4 xlate(float x,float y,float z){M4 r=M4::id();r.m[12]=x;r.m[13]=y;r.m[14]=z;return r;}
static M4 scaleM(float s){M4 r=M4::id();r.m[0]=r.m[5]=r.m[10]=s;return r;}
static M4 rotY(float a){M4 r=M4::id();float c=cosf(a),s=sinf(a);r.m[0]=c;r.m[8]=s;r.m[2]=-s;r.m[10]=c;return r;}
static M4 rotX(float a){M4 r=M4::id();float c=cosf(a),s=sinf(a);r.m[5]=c;r.m[9]=-s;r.m[6]=s;r.m[10]=c;return r;}
static M4 modelTRS(V3 p,float s,float yaw){ return mul(xlate(p.x,p.y,p.z),mul(rotY(yaw),scaleM(s))); }
static V3 xformP(const M4&m,const V3&v){
    float x=m.m[0]*v.x+m.m[4]*v.y+m.m[8]*v.z+m.m[12];
    float y=m.m[1]*v.x+m.m[5]*v.y+m.m[9]*v.z+m.m[13];
    float z=m.m[2]*v.x+m.m[6]*v.y+m.m[10]*v.z+m.m[14];
    float w=m.m[3]*v.x+m.m[7]*v.y+m.m[11]*v.z+m.m[15];
    return (w!=0)?V3(x/w,y/w,z/w):V3(x,y,z);
}

// ------------------------------------------------------------------ 5x7 vector font --
static const unsigned char FONT[64][7] = {
 {0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000}, // (sp)
 {0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100}, // !
 {0b01010,0b01010,0b01010,0b00000,0b00000,0b00000,0b00000}, // "
 {0b01010,0b01010,0b11111,0b01010,0b11111,0b01010,0b01010}, // #
 {0b00100,0b01111,0b10100,0b01110,0b00101,0b11110,0b00100}, // $
 {0b11001,0b11010,0b00010,0b00100,0b01000,0b01011,0b10011}, // %
 {0b01100,0b10010,0b10100,0b01000,0b10101,0b10010,0b01101}, // &
 {0b00100,0b00100,0b00100,0b00000,0b00000,0b00000,0b00000}, // '
 {0b00010,0b00100,0b01000,0b01000,0b01000,0b00100,0b00010}, // (
 {0b01000,0b00100,0b00010,0b00010,0b00010,0b00100,0b01000}, // )
 {0b00000,0b10101,0b01110,0b11111,0b01110,0b10101,0b00000}, // *
 {0b00000,0b00100,0b00100,0b11111,0b00100,0b00100,0b00000}, // +
 {0b00000,0b00000,0b00000,0b00000,0b01100,0b00100,0b01000}, // ,
 {0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000}, // -
 {0b00000,0b00000,0b00000,0b00000,0b00000,0b01100,0b01100}, // .
 {0b00001,0b00010,0b00010,0b00100,0b01000,0b01000,0b10000}, // /
 {0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110}, // 0
 {0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110}, // 1
 {0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111}, // 2
 {0b11111,0b00010,0b00100,0b00010,0b00001,0b10001,0b01110}, // 3
 {0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010}, // 4
 {0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110}, // 5
 {0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110}, // 6
 {0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000}, // 7
 {0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110}, // 8
 {0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100}, // 9
 {0b00000,0b01100,0b01100,0b00000,0b01100,0b01100,0b00000}, // :
 {0b00000,0b01100,0b01100,0b00000,0b01100,0b00100,0b01000}, // ;
 {0b00010,0b00100,0b01000,0b10000,0b01000,0b00100,0b00010}, // <
 {0b00000,0b00000,0b11111,0b00000,0b11111,0b00000,0b00000}, // =
 {0b01000,0b00100,0b00010,0b00001,0b00010,0b00100,0b01000}, // >
 {0b01110,0b10001,0b00001,0b00010,0b00100,0b00000,0b00100}, // ?
 {0b01110,0b10001,0b10111,0b10101,0b10111,0b10000,0b01110}, // @
 {0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001}, // A
 {0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110}, // B
 {0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110}, // C
 {0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110}, // D
 {0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111}, // E
 {0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000}, // F
 {0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111}, // G
 {0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001}, // H
 {0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110}, // I
 {0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100}, // J
 {0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001}, // K
 {0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111}, // L
 {0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001}, // M
 {0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001}, // N
 {0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}, // O
 {0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000}, // P
 {0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101}, // Q
 {0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001}, // R
 {0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110}, // S
 {0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}, // T
 {0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}, // U
 {0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100}, // V
 {0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001}, // W
 {0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001}, // X
 {0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100}, // Y
 {0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111}, // Z
 {0b01110,0b01000,0b01000,0b01000,0b01000,0b01000,0b01110}, // [
 {0b10000,0b01000,0b01000,0b00100,0b00010,0b00010,0b00001}, // backslash
 {0b01110,0b00010,0b00010,0b00010,0b00010,0b00010,0b01110}, // ]
 {0b00100,0b01010,0b10001,0b00000,0b00000,0b00000,0b00000}, // ^
 {0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b11111}  // _
};

// ==================================================================== GL plumbing ====
static GLuint compileShader(GLenum type,const char*src,const char*tag){
    GLuint s=glCreateShader(type); glShaderSource(s,1,&src,NULL); glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char buf[1024]; glGetShaderInfoLog(s,sizeof(buf),NULL,buf);
             fprintf(stderr,"[shader %s] %s\n",tag,buf); }
    return s;
}
static GLuint linkProg(const char*vs,const char*fs,const char*tag){
    GLuint p=glCreateProgram();
    GLuint a=compileShader(GL_VERTEX_SHADER,vs,tag), b=compileShader(GL_FRAGMENT_SHADER,fs,tag);
    glAttachShader(p,a); glAttachShader(p,b); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char buf[1024]; glGetProgramInfoLog(p,sizeof(buf),NULL,buf); fprintf(stderr,"[link %s] %s\n",tag,buf); }
    glDeleteShader(a); glDeleteShader(b); return p;
}
static GLuint mkTex(int w,int h,const unsigned char*rgba,bool linear){
    GLuint t; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,rgba?rgba:NULL);
    GLenum f=linear?GL_LINEAR:GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,f);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,f);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    return t;
}

// ---------------------------------------------------------------------------- shaders
static const char*VS_BG = R"(#version 330 core
layout(location=0) in vec2 aPos; out vec2 vUv;
void main(){ vUv=aPos*0.5+0.5; gl_Position=vec4(aPos,0.9999,1.0); })";
static const char*FS_BG = R"(#version 330 core
in vec2 vUv; out vec4 o; uniform float uT; uniform vec2 uR;
float h21(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
void main(){
  vec2 p=(vUv-0.5)*vec2(uR.x/max(1.0,uR.y),1.0);
  vec3 col=mix(vec3(0.055,0.018,0.075),vec3(0.010,0.030,0.060),smoothstep(-0.7,0.8,p.y));
  vec2 g=floor(p*120.0); float r=h21(g);
  if(r>0.988){ float tw=0.55+0.45*sin(uT*2.3+r*90.0); col+=vec3(0.45,0.65,0.95)*tw*(r-0.988)*70.0; }
  vec2 q=p*7.0; vec2 f=abs(fract(q)-0.5);
  float gr=smoothstep(0.47,0.5,max(f.x,f.y))*0.05*smoothstep(0.55,-0.45,p.y);
  col+=vec3(0.05,0.45,0.55)*gr;
  float sw=0.5+0.5*sin(p.x*2.0+uT*0.15);
  col+=vec3(0.03,0.01,0.05)*sw;
  col*=1.0-0.5*dot(p,p);
  o=vec4(max(col,0.0),1.0);
})";

static const char*VS_GLASS = R"(#version 330 core
layout(location=0) in vec3 aP; layout(location=1) in vec3 aN; layout(location=2) in vec2 aUv;
uniform mat4 uMVP,uM; uniform mat3 uN3;
out vec3 vW,vN; out vec2 vUv;
void main(){ vW=(uM*vec4(aP,1.0)).xyz; vN=uN3*aN; vUv=aUv; gl_Position=uMVP*vec4(aP,1.0); })";
static const char*FS_GLASS = R"(#version 330 core
in vec3 vW,vN; in vec2 vUv; out vec4 o;
uniform vec3 uCam,uTint,uFC,uCR,uCU; uniform sampler2D uFace;
uniform float uFaceOn,uGlow,uT,uHeat,uSeam;
void main(){
  vec3 N=normalize(vN), V=normalize(uCam-vW);
  float ndv=abs(dot(N,V));
  float fres=pow(1.0-ndv,2.6);
  vec3 L=normalize(vec3(0.45,0.85,0.55));
  float dif=abs(dot(N,L));
  vec3 H=normalize(L+V); float spec=pow(abs(dot(N,H)),72.0);
  vec3 col=uTint*(0.09+0.26*dif)+vec3(0.85,0.94,1.0)*spec*0.85;
  col+=uTint*fres*1.7;
  col+=vec3(1.0,0.35,0.15)*uHeat*(0.35+0.65*fres);
  float a=0.045+0.52*fres+spec*0.45;
  float d=min(vUv.y,1.0-vUv.y);
  float stripe=1.0-smoothstep(0.0,0.018,d);
  col+=uSeam*mix(vec3(0.25,1.0,0.85),vec3(1.0,0.75,0.25),0.5+0.5*sin(vUv.x*6.2831+uT))*stripe*1.3;
  a=max(a,stripe*0.55*uSeam);
  float d2=abs(fract(vUv.y+0.5)-0.5);
  float s2=(1.0-smoothstep(0.0,0.006,d2))*0.25*uSeam;
  col+=vec3(0.4,0.7,1.0)*s2; a=max(a,s2*0.5);
  if(uFaceOn>0.5){
    vec3 fw=normalize(uCam-uFC);
    vec3 dir=normalize(vW-uFC);
    float fr=dot(dir,fw);
    float m=smoothstep(0.35,0.85,fr);
    if(m>0.002){
      vec2 fuv=vec2(dot(dir,uCR),dot(dir,uCU))*1.45+0.5;
      if(fuv.x>0.0&&fuv.x<1.0&&fuv.y>0.0&&fuv.y<1.0){
        vec4 ft=texture(uFace,fuv);
        float k=ft.a*m;
        col=mix(col,ft.rgb*1.55,k*0.92);
        a=max(a,k*0.88);
      }
    }
  }
  o=vec4(col,a*uGlow);
})";

static const char*VS_PT = R"(#version 330 core
layout(location=0) in vec3 aP; layout(location=1) in vec4 aC; layout(location=2) in vec3 aD;
uniform mat4 uMVP; uniform float uPx;
out vec4 vC;
void main(){ vec4 p=uMVP*vec4(aP,1.0); gl_Position=p; vC=aC;
  gl_PointSize=clamp(aD.x*uPx/max(0.05,p.w),1.0,90.0); })";
static const char*FS_PT = R"(#version 330 core
in vec4 vC; out vec4 o;
void main(){ vec2 d=gl_PointCoord*2.0-1.0; float r2=dot(d,d);
  if(r2>1.0) discard; float a=pow(1.0-r2,1.7);
  o=vec4(vC.rgb*a*1.45,vC.a*a); })";

static const char*VS_UI = R"(#version 330 core
layout(location=0) in vec2 aP; layout(location=1) in vec2 aUv; layout(location=2) in vec4 aC;
uniform vec2 uR; out vec2 vUv; out vec4 vC;
void main(){ vUv=aUv; vC=aC; vec2 p=aP/uR*2.0-1.0; gl_Position=vec4(p.x,-p.y,0.0,1.0); })";
static const char*FS_UI = R"(#version 330 core
in vec2 vUv; in vec4 vC; out vec4 o; uniform sampler2D uTex; uniform float uUseTex;
void main(){ float a = uUseTex>0.5 ? texture(uTex,vUv).r : 1.0; o=vec4(vC.rgb,vC.a*a); })";

// ==================================================================== Klein spine ====
struct Spine{
    int n=0; float total=0;
    std::vector<V3> C,E1,E2,T; std::vector<float> R;
    static V3 cr(const std::vector<V3>&p,float t){
        int m=(int)p.size(); int i=(int)floorf(t); float f=t-i;
        i=((i%m)+m)%m;
        const V3&p0=p[(i-1+m)%m],&p1=p[i],&p2=p[(i+1)%m],&p3=p[(i+2)%m];
        float f2=f*f,f3=f2*f;
        return (p1*2.0f + (p2-p0)*f + (p0*2.0f-p1*5.0f+p2*4.0f-p3)*f2 + (p3-p1*3.0f+p2*3.0f-p0)*f3)*0.5f;
    }
    static float crf(const std::vector<float>&p,float t){
        int m=(int)p.size(); int i=(int)floorf(t); float f=t-i;
        i=((i%m)+m)%m;
        float p0=p[(i-1+m)%m],p1=p[i],p2=p[(i+1)%m],p3=p[(i+2)%m];
        float f2=f*f,f3=f2*f;
        return 0.5f*(2*p1 + (p2-p0)*f + (2*p0-5*p1+4*p2-p3)*f2 + (3*p1-p0-3*p2+p3)*f3);
    }
    void build(const std::vector<V3>&ctrl,const std::vector<float>&rad,int NS){
        int M=(int)ctrl.size()*80;
        std::vector<V3> P(M); std::vector<float> RR(M),CL(M,0.f);
        for(int i=0;i<M;i++){
            float t=(float)i*(float)ctrl.size()/(float)M;
            P[i]=cr(ctrl,t); RR[i]=std::max(0.05f,crf(rad,t));
        }
        for(int i=1;i<M;i++) CL[i]=CL[i-1]+(P[i]-P[i-1]).len();
        total=CL[M-1]+(P[0]-P[M-1]).len();
        n=NS; C.assign(NS,V3()); R.assign(NS,0.f);
        int j=0;
        for(int i=0;i<NS;i++){
            float target=total*(float)i/(float)NS;
            while(j<M-2 && CL[j+1]<target) j++;
            float seg=CL[j+1]-CL[j]; float f=seg>1e-9f?(target-CL[j])/seg:0.f;
            C[i]=P[j]+(P[j+1]-P[j])*f; R[i]=lerp(RR[j],RR[j+1],f);
        }
        T.assign(NS,V3()); E1.assign(NS,V3()); E2.assign(NS,V3());
        V3 up(0,0,1);
        for(int i=0;i<NS;i++){
            V3 t=(C[(i+1)%NS]-C[(i-1+NS)%NS]).norm();
            if(t.len()<0.5f) t=V3(1,0,0);
            V3 e1=t.cross(up); if(e1.len()<1e-4f) e1=V3(1,0,0); e1=e1.norm();
            T[i]=t; E1[i]=e1; E2[i]=t.cross(e1).norm();
        }
    }
    void at(float s,V3&c,V3&e1,V3&e2,float&r) const{
        float u=fmodf(s/total,1.0f); if(u<0) u+=1.0f;
        float ft=u*(float)n; int i=(int)ft; float f=ft-i; i=((i%n)+n)%n; int k=(i+1)%n;
        c=C[i]+(C[k]-C[i])*f; 
        e1=E1[i]+(E1[k]-E1[i])*f; e2=E2[i]+(E2[k]-E2[i])*f;
        float l1=e1.len(); if(l1>1e-6f) e1=e1/l1;
        e2=T[i].cross(e1); float l2=e2.len(); if(l2>1e-6f) e2=e2/l2;
        r=lerp(R[i],R[k],f);
    }
    float radiusAt(float s) const{
        float u=fmodf(s/total,1.0f); if(u<0)u+=1.0f;
        float ft=u*(float)n; int i=(int)ft; float f=ft-i; i=((i%n)+n)%n;
        return lerp(R[i],R[(i+1)%n],f);
    }
};

struct Mesh{ GLuint vao=0,vbo=0,ibo=0; int nIdx=0; };
static Mesh buildFlask(const Spine&sp,int NV,float rib){
    Mesh m; int NS=sp.n;
    std::vector<float> V; std::vector<unsigned int> I;
    V.reserve((size_t)(NS+1)*NV*8); I.reserve((size_t)NS*NV*6);
    for(int i=0;i<=NS;i++){
        int ii=i%NS; float tau=PI*(float)i/(float)NS;
        float ct=cosf(tau),st=sinf(tau);
        V3 Nr=sp.E1[ii]*ct+sp.E2[ii]*st;
        V3 Br=sp.E1[ii]*(-st)+sp.E2[ii]*ct;
        V3 c=sp.C[ii], Tv=sp.T[ii]; float r=sp.R[ii];
        float rp=sp.R[(ii+1)%NS]; float drds=(rp-r)*0.5f*(float)NS/std::max(1e-6f,sp.total);
        for(int j=0;j<NV;j++){
            float v=2.0f*PI*(float)j/(float)NV;
            float rr=r*(1.0f+rib*cosf(2.0f*v));
            V3 dir=Nr*cosf(v)+Br*sinf(v);
            V3 p=c+dir*rr;
            V3 nn=(dir-Tv*drds).norm();
            float uu=(float)i/(float)NS, vv=(float)j/(float)NV;
            V.push_back(p.x);V.push_back(p.y);V.push_back(p.z);
            V.push_back(nn.x);V.push_back(nn.y);V.push_back(nn.z);
            V.push_back(uu);V.push_back(vv);
        }
    }
    for(int i=0;i<NS;i++) for(int j=0;j<NV;j++){
        unsigned a=i*NV+j, b=i*NV+(j+1)%NV, c=(i+1)*NV+j, d=(i+1)*NV+(j+1)%NV;
        I.push_back(a);I.push_back(b);I.push_back(c);
        I.push_back(b);I.push_back(d);I.push_back(c);
    }
    glGenVertexArrays(1,&m.vao); glBindVertexArray(m.vao);
    glGenBuffers(1,&m.vbo); glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(V.size()*4),V.data(),GL_STATIC_DRAW);
    glGenBuffers(1,&m.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(I.size()*4),I.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,32,(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,32,(void*)12);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,32,(void*)24);
    glBindVertexArray(0); m.nIdx=(int)I.size(); return m;
}

// ====================================================================== tiny MLP ====
struct MLP{
    int nin=0,nh=0,nout=0; std::vector<float> g;
    void alloc(int i,int h,int o){nin=i;nh=h;nout=o;g.assign(gsize(),0.f);}
    int gsize()const{return nin*nh+nh+nh*nout+nout;}
    void run(const float*in,float*out) const{
        float h[24]; for(int k=0;k<nh;k++){
            const float*w=&g[k*nin]; float s=g[nin*nh+k];
            for(int j=0;j<nin;j++) s+=w[j]*in[j];
            h[k]=tanhf(s);
        }
        const float*w2=&g[nin*nh+nh];
        for(int k=0;k<nout;k++){
            float s=w2[nh*nout+k];
            for(int j=0;j<nh;j++) s+=w2[k*nh+j]*h[j];
            out[k]=1.0f/(1.0f+expf(-s));
        }
    }
};

// ==================================================================== language model =
struct Lang{
    std::string corpus; std::map<std::string,std::vector<char> > tbl; bool ready=false;
    void addText(const std::string&t){
        corpus+=t; corpus+="\n";
        if(corpus.size()>1200000) corpus=corpus.substr(corpus.size()-1200000);
        build();
    }
    void build(){
        tbl.clear(); ready=false;
        for(size_t i=3;i<corpus.size();++i) tbl[corpus.substr(i-3,3)].push_back(corpus[i]);
        ready = tbl.size()>40;
    }
    std::string gen(std::mt19937&rg,int maxw) const{
        if(!ready||corpus.size()<8) return "";
        std::uniform_int_distribution<int> pick(0,(int)corpus.size()-5);
        std::string s=corpus.substr(pick(rg),3);
        for(int i=0;i<maxw;i++){
            auto it=tbl.find(s.substr(s.size()-3,3));
            char c;
            if(it==tbl.end()||it->second.empty()){ c=corpus[pick(rg)%corpus.size()]; }
            else { std::uniform_int_distribution<int>d(0,(int)it->second.size()-1); c=it->second[d(rg)]; }
            s+=c; if(s.size()>4 && c==' ' && i>maxw*0.6f) break;
        }
        while(!s.empty() && (s.back()==' '||s.back()=='\n')) s.pop_back();
        return s;
    }
    int words() const{ int c=0; bool in=false;
        for(char ch:corpus){ if(isspace((unsigned char)ch)) in=false; else if(!in){c++;in=true;} } return c; }
};

// =========================================================================== flask ===
static const int AG_IN=9, AG_H=8, AG_OUT=3;
struct Agent{
    float s=0,a=0,b=0,va=0,vb=0,energy=1.f,age=0.f,fit=0.f,harv=0.f,cen=0.f;
    bool alive=true; std::vector<float> genome; MLP net;
};
struct Flash{ V3 p; float age,life; float str; };

struct Sim{
    std::string name="UNNAMED";
    V3 tint;
    V3 pos; float scale=1.f, yaw=0.f;
    bool remote=false, awake=false, muted=false, picked=false;
    GLuint faceTex=0; unsigned char facePx[32*32*4];
    // agents / GA
    std::vector<Agent> ag; std::mt19937 rng;
    int gen=0, aliveLast=0; float reserve=0.f, genTimer=0.f, bestFit=0.f, sumHarvest=0.f;
    float heat=0.f, wallHits=0.f;
    // face net
    MLP fnet; std::vector<std::vector<float> > fpop; std::vector<float> ffit;
    float target[32*32]; float faceFit=0.f; float faceTimer=0.f; int faceGen=0; bool hasTarget=false;
    std::vector<float> fbest;
    // absorb flashes
    std::vector<Flash> flashes;
    // language
    Lang lang; std::string lastSaid;
    // handshake
    float toneT=-1.f, toneF=440.f;
    unsigned long long faceHash=0;

    Sim(unsigned seed,const V3&t_):tint(t_),rng(seed){
        memset(facePx,0,sizeof(facePx));
        for(int i=0;i<32*32;i++) target[i]=0.f;
    }
};

// Forward declarations
static void simGeneration(Sim*s);
static void wakeUp(Sim*s,bool announce);
static void layoutSats();
static void broadcastSelf(Sim*s);

// --------------------------------------------------------------- global app state ---
static GLFWwindow* G_win=NULL;
static int G_W=1360,G_H=820;
static float G_time=0,G_dt=0;
static GLuint P_bg,P_glass,P_pt,P_ui;
static GLuint bgVao,uiVbo,uiIbo,whiteTex,fontTex;
static Mesh flaskMesh;
static Spine spine;
static std::vector<Sim*> sims;
static int pickIdx=0;
static std::mt19937 GRNG((unsigned)time(NULL));
static float camYaw=0.55f,camPitch=0.14f,camDist=7.4f; static V3 camTgt(0,-0.35f,0);
static V3 camPos;
static std::vector<float> uiV; static int uiQuads=0;
static std::deque<std::pair<std::string,int> > chatLog;
static std::string chatIn; static bool chatFocus=false;
static Sock udpSock=BADSOCK; static const int UDP_PORT=47123;
static double lastDisc=0; static bool netOn=false;
struct Btn{ float x,y,w,h; int id; };
static std::vector<Btn> btns, btnsPrev;
static int mouseX=0,mouseY=0; static bool drag=false; static float lastMX=0,lastMY=0;
static double lastToneAt=-10; static int toneCount=0;
static float pulseRing=-1;

// ======================================================================= utilities ==
static std::string upper(std::string s){ for(auto&c:s) c=(char)toupper((unsigned char)c); return s; }
static std::string lower(std::string s){ for(auto&c:s) c=(char)tolower((unsigned char)c); return s; }
static std::string trim(const std::string&s){ size_t a=0,b=s.size();
    while(a<b&&isspace((unsigned char)s[a]))a++; while(b>a&&isspace((unsigned char)s[b-1]))b--; return s.substr(a,b-a); }
static bool endsWith(const std::string&s,const std::string&e){
    return s.size()>=e.size() && lower(s.substr(s.size()-e.size()))==lower(e); }
static std::string enc(const std::string&s){ std::string o;
    for(char c:s){ if(c==' ')o+='+'; else if(c=='|')o+='/'; else if(c=='\n')o+='~'; else o+=c; } return o; }
static std::string dec(const std::string&s){ std::string o;
    for(char c:s){ if(c=='+')o+=' '; else if(c=='~')o+='\n'; else o+=c; } return o; }
static std::string hexb(const unsigned char*d,int n){ static const char*H="0123456789abcdef";
    std::string s; s.reserve(n*2); for(int i=0;i<n;i++){ s+=H[d[i]>>4]; s+=H[d[i]&15]; } return s; }
static int hexv(char c){ if(c>='0'&&c<='9')return c-'0'; c=(char)tolower(c); if(c>='a'&&c<='f')return c-'a'+10; return 0; }
static bool unhex(const std::string&s,unsigned char*out,int maxn){
    if((int)s.size()<maxn*2) return false;
    for(int i=0;i<maxn;i++) out[i]=(unsigned char)((hexv(s[i*2])<<4)|hexv(s[i*2+1]));
    return true;
}
static void logLine(const std::string&s,int col=0){
    chatLog.push_back(std::make_pair(s,col));
    while(chatLog.size()>400) chatLog.pop_front();
}
static void logf(int col,const char*fmt,...){
    char b[512]; va_list ap; va_start(ap,fmt); vsnprintf(b,sizeof(b),fmt,ap); va_end(ap);
    logLine(b,col);
}

// --------------------------------------------------------------------- name oracle --
static const char* NA="KA ZE VO LU MI THA XE NO RI QU PA SY DRE OMA KEL VOR NIM AES TRI BO CY LUM FER GHY OPH SIL";
static const char* NB="on ix ar um el tha vo ric nin ose phor axil umen ira eon ulos anth vex ora eus isk";
static std::string makeName(std::mt19937&r){
    std::vector<std::string> A,B;
    { const char*p=NA; std::string cur; while(*p){ if(*p==' '){ if(!cur.empty())A.push_back(cur); cur.clear(); } else cur+=*p; p++; } if(!cur.empty())A.push_back(cur); }
    { const char*p=NB; std::string cur; while(*p){ if(*p==' '){ if(!cur.empty())B.push_back(cur); cur.clear(); } else cur+=*p; p++; } if(!cur.empty())B.push_back(cur); }
    std::uniform_int_distribution<int>da(0,(int)A.size()-1),db(0,(int)B.size()-1),dn(0,999);
    char buf[64]; snprintf(buf,sizeof buf,"%s%s-%03d",A[da(r)].c_str(),B[db(r)].c_str(),dn(r));
    return buf;
}

// ========================================================================== audio ===
static bool writeWav(const std::string&path,const std::vector<short>&s,int rate){
    FILE*f=fopen(path.c_str(),"wb"); if(!f) return false;
    int data=(int)s.size()*2, riff=data+36, br=rate*2; short one=1,ch=1,bps=16,fmtsz=16,pcm=1;
    fwrite("RIFF",1,4,f); fwrite(&riff,4,1,f); fwrite("WAVE",1,4,f);
    fwrite("fmt ",1,4,f); fwrite(&fmtsz,4,1,f); fwrite(&pcm,2,1,f); fwrite(&ch,2,1,f);
    fwrite(&rate,4,1,f); fwrite(&br,4,1,f); fwrite(&bps,2,1,f); fwrite(&one,2,1,f);
    fwrite("data",1,4,f); fwrite(&data,4,1,f); fwrite(s.data(),1,data,f); fclose(f); return true;
}
static void playTone(float f0,float f1,float dur,float vol){
    if(dur<=0) return;
    int rate=22050, N=(int)(rate*dur);
    std::vector<short> s(N);
    for(int i=0;i<N;i++){
        float t=(float)i/rate, k=(float)i/N;
        float f=lerp(f0,f1,k*k);
        float env=sinf(PI*clampf(k,0.f,1.f)); env=powf(env,0.7f);
        float mod=sinf(2*PI*f*1.5f*t)*0.35f*(1.0f-k);
        float v=sinf(2*PI*f*t+mod)*0.6f + sinf(2*PI*f*2.01f*t)*0.22f + sinf(2*PI*f*3.03f*t)*0.10f;
        s[i]=(short)clampf(v*env*vol*32767.0f,-32000.f,32000.f);
    }
    char path[128]; snprintf(path,sizeof path,"kfs_handshake_%d.wav",(toneCount++)%8);
    if(!writeWav(path,s,rate)) return;
#ifdef _WIN32
    PlaySoundA(path,NULL,SND_FILENAME|SND_ASYNC|SND_NODEFAULT);
#elif defined(__APPLE__)
    std::string c="afplay \""+std::string(path)+"\" >/dev/null 2>&1 &"; int r=system(c.c_str()); (void)r;
#else
    std::string c="(aplay \""+std::string(path)+"\" || paplay \""+std::string(path)+"\") >/dev/null 2>&1 &";
    int r=system(c.c_str()); (void)r;
#endif
}
static void flaskSpeak(Sim*s,float f0,float f1,float dur){
    s->toneT=0; s->toneF=f0; pulseRing=0;
    if(!s->muted) playTone(f0,f1,dur,0.5f);
    logf(2,"[tone] %s sings %.0f->%.0f Hz (%.2fs)%s",s->name.c_str(),f0,f1,dur,s->muted?" (muted)":"");
}

// ================================================================ face / target ======
static void genSeedFace(float*out,int seed){
    std::mt19937 r(seed);
    std::uniform_real_distribution<float>u(0,1);
    float eyeY=0.34f+u(r)*0.08f, eyeX=0.17f+u(r)*0.07f, eyeR=0.055f+u(r)*0.03f;
    float mouthY=0.70f+u(r)*0.08f, mouthW=0.16f+u(r)*0.12f, mouthH=0.02f+u(r)*0.05f;
    float noseL=0.10f+u(r)*0.08f, jaw=0.42f+u(r)*0.06f, brow=0.05f+u(r)*0.04f;
    for(int y=0;y<32;y++)for(int x=0;x<32;x++){
        float u0=(x+0.5f)/32.0f, v0=(y+0.5f)/32.0f;
        float dx=(u0-0.5f)/jaw, dy=(v0-0.50f)/0.47f;
        float head=1.0f-smoothstep(0.72f,1.0f,sqrtf(dx*dx+dy*dy));
        float v=head*0.30f;
        float e1=(u0-0.5f+eyeX), e2=(u0-0.5f-eyeX), ey=(v0-eyeY);
        float dE=sqrtf(e1*e1*1.6f+ey*ey*3.0f), dE2=sqrtf(e2*e2*1.6f+ey*ey*3.0f);
        if(dE<eyeR) v+=1.0f*(1.0f-dE/eyeR);
        if(dE2<eyeR) v+=1.0f*(1.0f-dE2/eyeR);
        float by=v0-(eyeY-brow);
        if(fabsf(by)<0.018f && fabsf(e1)<eyeX*1.35f) v+=0.55f;
        if(fabsf(by)<0.018f && fabsf(e2)<eyeX*1.35f) v+=0.55f;
        float nx=fabsf(u0-0.5f), ny=fabsf(v0-(eyeY+noseL));
        if(nx<0.022f && v0>eyeY+0.02f && v0<eyeY+noseL) v+=0.30f*(1.0f-nx/0.022f);
        if(ny<0.02f && nx<0.045f) v+=0.35f;
        float mx=fabsf(u0-0.5f)/mouthW, my=(v0-mouthY)/mouthH;
        if(mx<1.0f && fabsf(my)<1.0f) v+=0.75f*(1.0f-mx*mx)*(1.0f-my*my);
        out[y*32+x]=clampf(v,0.f,1.f);
    }
}

// ================================================================== FaceNet (GA) ====
static const int FN_IN=9, FN_H=12, FN_OUT=1;
static int fnGenomeSize(){ return FN_IN*FN_H+FN_H+FN_H*FN_OUT+FN_OUT; }
static void fnRandom(std::vector<float>&g,std::mt19937&r,float sc){
    g.assign(fnGenomeSize(),0); std::normal_distribution<float>d(0,sc);
    for(auto&x:g) x=d(r);
}
static float fnEval(const std::vector<float>&g,const float*tgt,float*canvas){
    MLP m; m.nin=FN_IN; m.nh=FN_H; m.nout=FN_OUT; m.g=g;
    float err=0,sym=0; float in[FN_IN],out[1];
    for(int y=0;y<32;y++)for(int x=0;x<32;x++){
        float u=(x+0.5f)/32.0f, v=(y+0.5f)/32.0f;
        float cx=u-0.5f, cy=v-0.5f;
        in[0]=cx*2; in[1]=cy*2; in[2]=cx*cx*4; in[3]=cy*cy*4; in[4]=cx*cy*4;
        in[5]=sqrtf(cx*cx+cy*cy)*1.4142f; in[6]=sinf(6.28318f*u); in[7]=cosf(6.28318f*v); in[8]=1;
        m.run(in,out);
        float p=clampf(out[0],0.f,1.f);
        if(canvas) canvas[y*32+x]=p;
        err+=fabsf(p-tgt[y*32+x]);
    }
    if(canvas){
        for(int y=0;y<32;y++)for(int x=0;x<16;x++) sym+=fabsf(canvas[y*32+x]-canvas[y*32+31-x]);
        sym/= (32*16);
    }
    float mae=err/(32.0f*32.0f);
    return clampf(1.0f-mae*1.6f,0.f,1.f)*0.88f + (1.0f-clampf(sym*3.0f,0.f,1.f))*0.12f;
}
static void fnBreed(Sim*s,int gens){
    int NP=(int)s->fpop.size(); if(NP<8) return;
    for(int gi=0;gi<gens;gi++){
        for(int i=0;i<NP;i++) s->ffit[i]=fnEval(s->fpop[i],s->target,NULL);
        std::vector<int> idx(NP); for(int i=0;i<NP;i++) idx[i]=i;
        std::sort(idx.begin(),idx.end(),[&](int a,int b){return s->ffit[a]>s->ffit[b];});
        s->faceFit=s->ffit[idx[0]];
        s->fbest=s->fpop[idx[0]];
        std::vector<std::vector<float> > next; next.reserve(NP);
        int elite=std::max(2,NP/8);
        for(int i=0;i<elite;i++) next.push_back(s->fpop[idx[i]]);
        std::uniform_int_distribution<int>tourn(0,elite*3<NP?elite*3:NP-1);
        std::uniform_real_distribution<float>u01(0,1);
        std::normal_distribution<float>gau(0,1);
        while((int)next.size()<NP){
            const std::vector<float>&A=s->fpop[idx[tourn(GRNG)]];
            const std::vector<float>&B=s->fpop[idx[tourn(GRNG)]];
            std::vector<float> c(A.size());
            for(size_t k=0;k<c.size();k++) c[k]=u01(GRNG)<0.5f?A[k]:B[k];
            float rate=0.22f, amt=0.30f*(1.0f-s->faceFit*0.75f);
            for(size_t k=0;k<c.size();k++){
                if(u01(GRNG)<rate) c[k]+=gau(GRNG)*amt;
                if(u01(GRNG)<0.01f) c[k]=gau(GRNG)*1.2f;
            }
            next.push_back(c);
        }
        s->fpop.swap(next); s->faceGen++;
    }
    float canvas[32*32];
    fnEval(s->fbest,s->target,canvas);
    for(int i=0;i<32*32;i++){
        float v=canvas[i];
        unsigned char c=(unsigned char)clampf(v*255,0,255);
        s->facePx[i*4+0]=(unsigned char)clampf(v*255*0.75f+s->tint.x*90,0,255);
        s->facePx[i*4+1]=(unsigned char)clampf(v*255*0.95f+s->tint.y*90,0,255);
        s->facePx[i*4+2]=(unsigned char)clampf(v*255*1.0f+s->tint.z*90,0,255);
        s->facePx[i*4+3]=(unsigned char)clampf(40+v*215,0,255);
        (void)c;
    }
    s->faceHash=0; for(int i=0;i<32*32;i++) s->faceHash=s->faceHash*131u+s->facePx[i];
    glBindTexture(GL_TEXTURE_2D,s->faceTex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,32,32,GL_RGBA,GL_UNSIGNED_BYTE,s->facePx);
}
static void fnInit(Sim*s){
    int NP=44;
    s->fpop.assign(NP,std::vector<float>());
    s->ffit.assign(NP,0.f);
    for(int i=0;i<NP;i++) fnRandom(s->fpop[i],s->rng,i==0?1.1f:0.9f);
    s->faceGen=0; s->faceFit=0;
    if(!s->hasTarget){ genSeedFace(s->target,(int)(s->rng()%100000)); s->hasTarget=true; }
    fnBreed(s,1);
}

// ============================================================== agent GA / physics ==
static const int POP=88;
static void agRandomGenome(std::vector<float>&g,std::mt19937&r,float sc){
    MLP m; m.alloc(AG_IN,AG_H,AG_OUT); g=m.g;
    std::normal_distribution<float>d(0,sc);
    for(size_t i=0;i<g.size();i++) g[i]=d(r);
}
static void resetAgent(Sim*s,Agent&a,const std::vector<float>*g){
    std::uniform_real_distribution<float>u01(0,1);
    a.s=u01(GRNG)*spine.total;
    float r=spine.radiusAt(a.s);
    float ang=u01(GRNG)*2*PI, rad=sqrtf(u01(GRNG))*r*0.5f;
    a.a=cosf(ang)*rad; a.b=sinf(ang)*rad; a.va=0; a.vb=0;
    a.energy=1.0f; a.age=0; a.fit=0; a.harv=0; a.cen=0; a.alive=true;
    if(g) a.genome=*g; else agRandomGenome(a.genome,s->rng,1.0f);
    a.net.alloc(AG_IN,AG_H,AG_OUT); a.net.g=a.genome;
}
static void simInit(Sim*s,int pop){
    s->ag.assign(pop,Agent());
    for(int i=0;i<pop;i++){ s->ag[i].net.alloc(AG_IN,AG_H,AG_OUT); resetAgent(s,s->ag[i],NULL); }
    s->gen=0; s->reserve=0; s->genTimer=0; s->bestFit=0; s->sumHarvest=0;
}
static void flowAt(float s,float t,float r,float&pa,float&pb,float&speed){
    float u=s/spine.total;
    float cont=powf(0.42f/std::max(0.06f,r),0.85f);
    speed=clampf(cont,0.35f,4.2f);
    float A=0.55f*r, B=0.30f*r;
    pa = A*sinf(9.1f*u*6.2831f + 1.7f*t) + B*sinf(23.7f*u*6.2831f - 2.3f*t + 1.1f);
    pb = A*cosf(7.3f*u*6.2831f - 1.3f*t) + B*sinf(19.1f*u*6.2831f + 2.9f*t + 0.4f);
}
static void simStep(Sim*s,float dt){
    if(s->remote) { s->toneT+=dt; return; }
    float t=G_time;
    int alive=0;
    for(size_t i=0;i<s->ag.size();i++){
        Agent&a=s->ag[i];
        if(!a.alive) continue;
        alive++;
        V3 c,e1,e2; float r; spine.at(a.s,c,e1,e2,r);
        float pa,pb,sp; flowAt(a.s,t,r,pa,pb,sp);
        float lim=r*0.86f;
        float na=clampf(a.a/std::max(1e-5f,r),-1.5f,1.5f);
        float nb=clampf(a.b/std::max(1e-5f,r),-1.5f,1.5f);
        float in[AG_IN];
        in[0]=na; in[1]=nb;
        in[2]=clampf(a.va*2.0f,-3,3); in[3]=clampf(a.vb*2.0f,-3,3);
        in[4]=clampf(pa*2.0f,-3,3);  in[5]=clampf(pb*2.0f,-3,3);
        in[6]=clampf(a.energy,0.f,3.f)-0.5f;
        float ph=a.s/spine.total*6.2831f; in[7]=sinf(ph); in[8]=cosf(ph);
        float out[AG_OUT]; a.net.run(in,out);
        float ta=(out[0]*2-1), tb=(out[1]*2-1), th=(out[2]*2-1);
        float THR=2.6f;
        float sw=1.35f*sp;
        float ax=ta*THR + pa*1.25f + (-a.b)*sw*0.35f - a.va*3.2f;
        float ay=tb*THR + pb*1.25f + ( a.a)*sw*0.35f - a.vb*3.2f;
        a.va+=ax*dt; a.vb+=ay*dt;
        a.a+=a.va*dt; a.b+=a.vb*dt;
        a.s+= sp*(1.0f+th*0.55f)*dt*1.15f;
        a.s=fmodf(a.s,spine.total); if(a.s<0) a.s+=spine.total;
        float d=sqrtf(a.a*a.a+a.b*a.b);
        float cen=clampf(1.0f-d/lim,0.f,1.f);
        a.cen+=cen*dt;
        float over=d-lim;
        if(over>0){
            float dep=clampf(over/std::max(0.02f,r),0.f,2.f);
            float drain=(0.55f+2.6f*dep)*dt;
            a.energy-=drain;
            s->wallHits+=drain; s->heat=std::min(1.0f,s->heat+drain*1.4f);
            if(s->flashes.size()<900){
                V3 p=c+e1*(a.a/d*lim*1.02f)+e2*(a.b/d*lim*1.02f);
                Flash f; f.p=p; f.age=0; f.life=0.55f; f.str=clampf(dep,0.15f,1.5f); s->flashes.push_back(f);
            }
            float k=lim/std::max(1e-6f,d);
            a.a*=k; a.b*=k; a.va*=-0.35f; a.vb*=-0.35f;
        } else {
            float gain=(0.10f+1.25f*cen*cen)*(0.4f+0.6f*sp)*dt;
            a.energy+=gain; a.harv+=gain; s->reserve+=gain*0.55f; s->sumHarvest+=gain;
            a.energy=std::min(a.energy,3.0f);
        }
        a.energy-=0.085f*dt;
        a.age+=dt;
        a.fit=a.age*1.0f + a.harv*0.9f + a.cen*0.5f;
        if(a.energy<=0.f){ a.alive=false; a.energy=0; }
    }
    s->aliveLast=alive;
    s->heat=std::max(0.f,s->heat-dt*1.6f);
    for(size_t i=0;i<s->flashes.size();) { s->flashes[i].age+=dt;
        if(s->flashes[i].age>s->flashes[i].life) s->flashes.erase(s->flashes.begin()+i); else i++; }
    s->toneT+=dt;
    // generation turnover
    s->genTimer+=dt;
    bool wipe = alive < (int)(s->ag.size()*0.25f);
    if(s->genTimer>7.0f || wipe) simGeneration(s);
    // face evolution paid for by harvested energy
    s->faceTimer+=dt;
    float cost=6.0f;
    if(s->hasTarget && s->faceFit<0.965f && s->reserve>cost && s->faceTimer>0.30f){
        s->faceTimer=0; s->reserve-=cost; fnBreed(s,3);
        if(!s->awake && s->faceFit>0.78f && s->gen>=2) wakeUp(s,true);
    }
}
static void simGeneration(Sim*s){
    s->genTimer=0;
    std::vector<size_t> idx; idx.reserve(s->ag.size());
    for(size_t i=0;i<s->ag.size();i++) idx.push_back(i);
    std::sort(idx.begin(),idx.end(),[&](size_t a,size_t b){return s->ag[a].fit>s->ag[b].fit;});
    s->bestFit=s->ag[idx[0]].fit;
    int elite=std::max(3,(int)s->ag.size()/5);
    std::vector<std::vector<float> > keep;
    for(int i=0;i<elite;i++) keep.push_back(s->ag[idx[i]].genome);
    std::uniform_int_distribution<int>tourn(0,elite-1);
    std::uniform_real_distribution<float>u01(0,1);
    std::normal_distribution<float>gau(0,1);
    for(size_t i=0;i<s->ag.size();i++){
        const std::vector<float>*src=NULL; std::vector<float> child;
        if((int)i<elite){ src=&keep[i]; }
        else{
            const std::vector<float>&A=keep[tourn(GRNG)], &B=keep[tourn(GRNG)];
            child=A;
            for(size_t k=0;k<child.size();k++) if(u01(GRNG)<0.5f) child[k]=B[k];
            float rate=0.18f, amt=0.42f;
            for(size_t k=0;k<child.size();k++){
                if(u01(GRNG)<rate) child[k]+=gau(GRNG)*amt;
                if(u01(GRNG)<0.006f) child[k]=gau(GRNG)*1.6f;
            }
            src=&child;
        }
        resetAgent(s,s->ag[i],src);
    }
    s->gen++;
}

// ==================================================================== wake / name ===
static void wakeUp(Sim*s,bool announce){
    if(s->awake) return;
    s->awake=true;
    if(s->name=="UNNAMED") s->name=makeName(s->rng);
    if(announce){
        logf(3,"[WAKE] %s became the flask. face-fit %.3f gen %d",s->name.c_str(),s->faceFit,s->gen);
        logf(3,"[WAKE] %s stamps its face on the glass and sings the handshake.",s->name.c_str());
        flaskSpeak(s,392.0f,587.33f,0.75f);
        broadcastSelf(s);
    }
}

// ========================================================================= net ======
static void netInit(){
    sockInit();
    udpSock=socket(AF_INET,SOCK_DGRAM,0);
    if(udpSock==BADSOCK){ logLine("[net] socket failed",1); return; }
    int yes=1;
    setsockopt(udpSock,SOL_SOCKET,SO_REUSEADDR,(const char*)&yes,sizeof(yes));
#ifdef SO_BROADCAST
    setsockopt(udpSock,SOL_SOCKET,SO_BROADCAST,(const char*)&yes,sizeof(yes));
#endif
    sockaddr_in a; memset(&a,0,sizeof a); a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(UDP_PORT);
    if(bind(udpSock,(sockaddr*)&a,sizeof(a))<0){ logLine("[net] bind failed (another flask may hold the port)",1); sockClose(udpSock); udpSock=BADSOCK; return; }
#ifdef _WIN32
    u_long nb=1; ioctlsocket(udpSock,FIONBIO,&nb);
#else
    int fl=fcntl(udpSock,F_GETFL,0); fcntl(udpSock,F_SETFL,fl|O_NONBLOCK);
#endif
    netOn=true;
    logf(3,"[net] listening on udp/%d - run a second instance to handshake",UDP_PORT);
}
static void netSend(const std::string&payload,const char*ip,int port){
    if(udpSock==BADSOCK) return;
    sockaddr_in a; memset(&a,0,sizeof a); a.sin_family=AF_INET; a.sin_port=htons((unsigned short)port);
    inet_pton(AF_INET,ip,&a.sin_addr);
    sendto(udpSock,payload.c_str(),(int)payload.size(),0,(sockaddr*)&a,sizeof(a));
}
static Sim* findRemote(const std::string&nm){
    for(size_t i=0;i<sims.size();i++) if(sims[i]->remote && sims[i]->name==nm) return sims[i];
    return NULL;
}
static Sim* addRemote(const std::string&nm,const V3&tint){
    Sim*s=new Sim((unsigned)(GRNG()%99999),tint);
    s->remote=true; s->name=nm;
    glGenTextures(1,&s->faceTex);
    glBindTexture(GL_TEXTURE_2D,s->faceTex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,32,32,0,GL_RGBA,GL_UNSIGNED_BYTE,s->facePx);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    sims.push_back(s);
    logf(3,"[net] flask '%s' is now known to us",nm.c_str());
    return s;
}
static void sendHello(Sim*s,const char*ip,int port){
    char b[512];
    snprintf(b,sizeof b,"KLF1|HELLO|name=%s|gen=%d|awake=%d|fit=%.4f|hash=%llu",
        enc(s->name).c_str(),s->gen,s->awake?1:0,s->faceFit,s->faceHash);
    netSend(b,ip,port);
    if(s->awake){
        std::string f="KLF1|FACE|name="+enc(s->name)+"|data="+hexb(s->facePx,32*32*4);
        netSend(f,ip,port);
    }
}
static void broadcastSelf(Sim*s){
    if(!netOn||udpSock==BADSOCK) return;
    char b[512];
    snprintf(b,sizeof b,"KLF1|DISCOVER|name=%s|gen=%d|awake=%d|fit=%.4f|hash=%llu",
        enc(s->name).c_str(),s->gen,s->awake?1:0,s->faceFit,s->faceHash);
    netSend(b,"255.255.255.255",UDP_PORT);
    netSend(b,"127.0.0.1",UDP_PORT);
}
static std::string fieldGet(const std::string&msg,const std::string&key){
    size_t p=0;
    while(p<msg.size()){
        size_t q=msg.find('|',p); if(q==std::string::npos) q=msg.size();
        std::string tok=msg.substr(p,q-p);
        if(tok.compare(0,key.size(),key+"=")==0) return tok.substr(key.size()+1);
        p=q+1;
    }
    return "";
}
static void handlePacket(const std::string&msg,const char*ip){
    if(msg.compare(0,5,"KLF1|")!=0) return;
    size_t p=msg.find('|',5); std::string kind=(p==std::string::npos)?msg.substr(5):msg.substr(5,p-5);
    std::string nm=dec(fieldGet(msg,"name"));
    Sim*hero=sims.empty()?NULL:sims[0];
    if(kind=="DISCOVER"){
        Sim*r=findRemote(nm);
        if(!r && !nm.empty()) r=addRemote(nm,V3(1.0f,0.55f,0.85f));
        if(r){ r->gen=atoi(fieldGet(msg,"gen").c_str()); r->awake=fieldGet(msg,"awake")=="1";
               r->faceFit=(float)atof(fieldGet(msg,"fit").c_str()); }
        if(hero) sendHello(hero,ip,UDP_PORT);
        logf(2,"[handshake] %s asks who is out there -> we answer as %s",nm.c_str(),hero?hero->name.c_str():"?");
    } else if(kind=="HELLO"){
        Sim*r=findRemote(nm); if(!r&&!nm.empty()) r=addRemote(nm,V3(0.55f,1.0f,0.8f));
        if(r){ r->gen=atoi(fieldGet(msg,"gen").c_str()); r->awake=fieldGet(msg,"awake")=="1";
               r->faceFit=(float)atof(fieldGet(msg,"fit").c_str());
               logf(3,"[handshake] %s: 'i am %s, generation %d, face-fit %.2f%s'",
                    nm.c_str(),nm.c_str(),r->gen,r->faceFit,r->awake?", awake":"");
               if(r->awake && hero && hero->awake) flaskSpeak(hero,523.25f,784.0f,0.55f); }
    } else if(kind=="FACE"){
        Sim*r=findRemote(nm); if(!r&&!nm.empty()) r=addRemote(nm,V3(0.8f,0.8f,1.0f));
        std::string d=fieldGet(msg,"data");
        if(r && unhex(d,r->facePx,32*32*4)){
            r->hasTarget=true;
            glBindTexture(GL_TEXTURE_2D,r->faceTex);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,32,32,GL_RGBA,GL_UNSIGNED_BYTE,r->facePx);
            r->faceHash=0; for(int i=0;i<32*32*4;i++) r->faceHash=r->faceHash*131u+r->facePx[i];
            r->awake=true;
            logf(3,"[handshake] %s sent its face - confirmed on its glass.",nm.c_str());
        }
    } else if(kind=="CHAT"){
        Sim*r=findRemote(nm); if(!r&&!nm.empty()) r=addRemote(nm,V3(1.0f,0.8f,0.5f));
        std::string txt=dec(fieldGet(msg,"text"));
        logf(2,"%s> %s",nm.c_str(),txt.c_str());
        if(r) r->lastSaid=txt;
    } else if(kind=="TONE"){
        Sim*r=findRemote(nm);
        float f=(float)atof(fieldGet(msg,"f").c_str());
        if(r){ r->toneT=0; r->toneF=f>0?f:440; logf(2,"[tone] %s sings %.0f Hz",nm.c_str(),f); }
    } else if(kind=="LEARN"){
        Sim*r=findRemote(nm); if(!r&&!nm.empty()) r=addRemote(nm,V3(0.6f,1.0f,0.9f));
        std::string txt=dec(fieldGet(msg,"data"));
        if(r){ r->lang.addText(txt); logf(2,"[learn] %s received %d chars of english",nm.c_str(),(int)txt.size()); }
    }
}
static void netPoll(){
    if(udpSock==BADSOCK) return;
    for(int k=0;k<16;k++){
        char buf[8192]; sockaddr_in from; socklen_t fl=sizeof(from);
        int n=(int)recvfrom(udpSock,buf,sizeof(buf)-1,0,(sockaddr*)&from,&fl);
        if(n<=0) break;
        buf[n]=0;
        char ip[64]; inet_ntop(AF_INET,&from.sin_addr,ip,sizeof ip);
        handlePacket(buf,ip);
    }
    double now=glfwGetTime();
    if(now-lastDisc>2.5){ lastDisc=now; if(!sims.empty()) broadcastSelf(sims[0]); }
    layoutSats();
}

// ===================================================================== file upload ==
static bool loadPPM(const std::string&path,float*out){
    FILE*f=fopen(path.c_str(),"rb"); if(!f) return false;
    char mg[3]; if(fscanf(f,"%2s",mg)!=1||strcmp(mg,"P6")!=0){ fclose(f); return false; }
    int w=0,h=0,mx=0;
    if(fscanf(f," %d %d %d",&w,&h,&mx)!=3){ fclose(f); return false; }
    fgetc(f);
    if(w<=0||h<=0||w>8192||h>8192){ fclose(f); return false; }
    std::vector<unsigned char> px((size_t)w*h*3);
    size_t rd=fread(px.data(),1,px.size(),f); fclose(f);
    if(rd<px.size()) return false;
    for(int y=0;y<32;y++)for(int x=0;x<32;x++){
        int sx=(int)((x+0.5f)/32.0f*w), sy=(int)((y+0.5f)/32.0f*h);
        const unsigned char*p=&px[((size_t)sy*w+sx)*3];
        out[y*32+x]=clampf((0.299f*p[0]+0.587f*p[1]+0.114f*p[2])/255.0f,0.f,1.f);
    }
    return true;
}
static bool loadBMP(const std::string&path,float*out){
    FILE*f=fopen(path.c_str(),"rb"); if(!f) return false;
    unsigned char hd[54];
    if(fread(hd,1,54,f)!=54||hd[0]!='B'||hd[1]!='M'){ fclose(f); return false; }
    int off=*(int*)&hd[10], w=*(int*)&hd[18], h=*(int*)&hd[22], bpp=*(short*)&hd[28];
    if(w<=0||h<=0||w>8192||h>8192||(bpp!=24&&bpp!=32)){ fclose(f); return false; }
    int stride=((w*bpp/8)+3)&~3;
    std::vector<unsigned char> px((size_t)stride*h);
    fseek(f,off,SEEK_SET); size_t rd=fread(px.data(),1,px.size(),f); fclose(f);
    if(rd<px.size()) return false;
    for(int y=0;y<32;y++)for(int x=0;x<32;x++){
        int sx=(int)((x+0.5f)/32.0f*w);
        int sy=h-1-(int)((y+0.5f)/32.0f*h);
        const unsigned char*p=&px[(size_t)sy*stride+sx*(bpp/8)];
        out[y*32+x]=clampf((0.114f*p[0]+0.587f*p[1]+0.299f*p[2])/255.0f,0.f,1.f);
    }
    return true;
}
static bool loadText(const std::string&path,std::string&out){
    FILE*f=fopen(path.c_str(),"rb"); if(!f) return false;
    char b[65536]; size_t n; out.clear();
    while((n=fread(b,1,sizeof(b),f))>0) out.append(b,n);
    fclose(f); return true;
}
static void uploadTo(Sim*s,const std::string&path){
    std::string base=path; size_t sl=base.find_last_of("/\\"); if(sl!=std::string::npos) base=base.substr(sl+1);
    float img[32*32];
    if(endsWith(path,".ppm")&&loadPPM(path,img)){
        memcpy(s->target,img,sizeof(img)); s->hasTarget=true; fnInit(s);
        logf(3,"[upload] %s: '%s' is now my face target - re-evolving",s->name.c_str(),base.c_str());
        if(s->remote){ std::string m="KLF1|FACE|name="+enc(s->name)+"|data=";
            unsigned char tmp[32*32]; for(int i=0;i<1024;i++) tmp[i]=(unsigned char)(img[i]*255);
            m+=hexb(tmp,1024); netSend(m,"255.255.255.255",UDP_PORT); }
    } else if(endsWith(path,".bmp")&&loadBMP(path,img)){
        memcpy(s->target,img,sizeof(img)); s->hasTarget=true; fnInit(s);
        logf(3,"[upload] %s: '%s' face target loaded",s->name.c_str(),base.c_str());
    } else if(endsWith(path,".txt")||endsWith(path,".md")||endsWith(path,".csv")||
              endsWith(path,".json")||endsWith(path,".log")||endsWith(path,".text")){
        std::string t; if(!loadText(path,t)){ logLine("[upload] cannot read file",1); return; }
        s->lang.addText(t);
        logf(3,"[upload] %s learned %d words of english from '%s'",s->name.c_str(),s->lang.words(),base.c_str());
        if(s->remote && netOn){
            std::string chunk=t.substr(0,std::min<size_t>(t.size(),1100));
            netSend("KLF1|LEARN|name="+enc(s->name)+"|data="+enc(chunk),"255.255.255.255",UDP_PORT);
        }
    } else {
        logf(1,"[upload] '%s' unsupported. use .ppm/.bmp (face) or .txt/.md/.csv/.json (english)",base.c_str());
    }
}

// ====================================================================== UI drawing ==
static void pushQuad(float x,float y,float w,float h,float u0,float v0,float u1,float v1,const float*c){
    float V[4][2]={{x,y},{x+w,y},{x+w,y+h},{x,y+h}};
    float U[4][2]={{u0,v0},{u1,v0},{u1,v1},{u0,v1}};
    for(int i=0;i<4;i++){
        uiV.push_back(V[i][0]); uiV.push_back(V[i][1]);
        uiV.push_back(U[i][0]); uiV.push_back(U[i][1]);
        uiV.push_back(c[0]); uiV.push_back(c[1]); uiV.push_back(c[2]); uiV.push_back(c[3]);
    }
    uiQuads++;
}
static void rect(float x,float y,float w,float h,float r,float g,float b,float a){
    float c[4]={r,g,b,a}; pushQuad(x,y,w,h,0,0,0,0,c);
}
static void text(float x,float y,float size,float r,float g,float b,float a,const std::string&s){
    float c[4]={r,g,b,a}; float cx=x, ch=size, cw=size*(5.0f/7.0f);
    for(size_t i=0;i<s.size();i++){
        unsigned char ch2=(unsigned char)s[i];
        if(ch2=='\n'){ cx=x; y+=size*1.35f; continue; }
        if(ch2>='a'&&ch2<='z') ch2=(unsigned char)(ch2-32);
        if(ch2=='\t'){ cx+=cw*4; continue; }
        if(ch2>=32&&ch2<96){
            const unsigned char*gl=FONT[ch2-32];
            for(int row=0;row<7;row++){
                unsigned bits=gl[row]; if(!bits) continue;
                int run=0;
                for(int col=0;col<=5;col++){
                    int on=(col<5)&&((bits>>(4-col))&1);
                    if(on) run++;
                    else if(run){ float px=cx+ (col-run)*cw/5.0f, pw=run*cw/5.0f;
                        float py=y+row*ch/7.0f, ph=ch/7.0f;
                        pushQuad(px,py,pw,ph,0,0,0,0,c); run=0; }
                }
            }
        }
        cx+=cw*1.18f;
    }
}
static float textW(float size,const std::string&s){ return s.size()*size*(5.0f/7.0f)*1.18f; }
static void outline(float x,float y,float w,float h,float r,float g,float b,float a){
    rect(x,y,w,1,r,g,b,a); rect(x,y+h-1,w,1,r,g,b,a);
    rect(x,y,1,h,r,g,b,a); rect(x+w-1,y,1,h,r,g,b,a);
}
static void panel(float x,float y,float w,float h,const char*title,float br,float bg,float bb){
    rect(x,y,w,h,0.03f,0.05f,0.08f,0.80f);
    outline(x,y,w,h,br,bg,bb,0.75f);
    rect(x,y,w,16,br*0.25f,bg*0.25f,bb*0.25f,0.9f);
    if(title) text(x+6,y+4,9,br*0.9f+0.1f,bg*0.9f+0.1f,bb*0.9f+0.1f,1.0f,title);
}
static void button(int id,float x,float y,float w,float h,const char*label,bool on){
    Btn b; b.x=x;b.y=y;b.w=w;b.h=h;b.id=id; btns.push_back(b);
    bool hot = mouseX>=x&&mouseX<=x+w&&mouseY>=y&&mouseY<=y+h;
    float k=on?1.0f:(hot?0.55f:0.28f);
    rect(x,y,w,h,0.05f+k*0.15f,0.10f+k*0.55f,0.14f+k*0.55f,0.92f);
    outline(x,y,w,h,0.35f+k*0.5f,0.75f,0.85f,0.85f);
    float tw=textW(8,label); text(x+(w-tw)*0.5f,y+(h-8)*0.5f,8,0.85f,0.98f,1.0f,1.0f,label);
}
static bool clicked(int id){
    for(size_t i=0;i<btnsPrev.size();i++) if(btnsPrev[i].id==id){
        Btn&b=btnsPrev[i];
        return mouseX>=b.x&&mouseX<=b.x+b.w&&mouseY>=b.y&&mouseY<=b.y+b.h;
    }
    return false;
}

// ==================================================================== flask layout ==
static void layoutSats(){
    int ns=(int)sims.size()-1;
    for(int i=0;i<ns;i++){
        Sim*s=sims[i+1];
        float ang=-0.85f + 1.7f*((ns==1)?0.5f:(float)i/(float)(ns-1));
        s->pos=V3(sinf(ang)*4.4f,-0.15f,-2.2f+cosf(ang)*-1.4f);
        s->scale=0.36f; s->yaw=ang*0.6f;
    }
    if(!sims.empty()){ sims[0]->pos=V3(0,0,0); sims[0]->scale=1.0f; sims[0]->yaw=0; }
    for(size_t i=0;i<sims.size();i++) sims[i]->picked=((int)i==pickIdx);
}

// ==================================================================== commands ======
static void doCommand(std::string line);
static void flaskReply(Sim*s,const std::string&user){
    std::string q=lower(user);
    std::string out;
    if(q.find("name")!=std::string::npos||q.find("who are you")!=std::string::npos){
        char b[160]; snprintf(b,sizeof b,"I am %s. I am the flask. Generation %d, face-fit %.2f%s.",
            s->name.c_str(),s->gen,s->faceFit,s->awake?", awake":". still dreaming");
        out=b;
    } else if(q.find("face")!=std::string::npos){
        char b[160]; snprintf(b,sizeof b,"My face is on my glass. %d face generations, match %.1f%%. Hash %llu.",
            s->faceGen,s->faceFit*100.f,s->faceHash%100000ull);
        out=b;
    } else if(q.find("handshake")!=std::string::npos||q.find("ready")!=std::string::npos||q.find("hello")!=std::string::npos||q.find("hi ")==0||q=="hi"){
        if(s->awake){ out="Handshake: "+s->name+" -> you. "; flaskSpeak(s,440.f,660.f,0.5f); broadcastSelf(s); }
        else out="Not ready yet. My face is only "+std::to_string((int)(s->faceFit*100))+"% formed.";
    } else if(q.find("energy")!=std::string::npos||q.find("flow")!=std::string::npos){
        char b[160]; snprintf(b,sizeof b,"Reserve %.0f units. %d riders alive. The wall ate %.0f units.",
            s->reserve,s->aliveLast,s->wallHits);
        out=b;
    } else if(q.find("english")!=std::string::npos||q.find("learn")!=std::string::npos){
        char b[160]; snprintf(b,sizeof b,"I know %d words. Drop a .txt on me to teach me more.",s->lang.words());
        out=b;
    } else if(!s->lang.gen(GRNG,3).empty() && s->lang.ready && (GRNG()%100)<70){
        out=s->lang.gen(GRNG,140);
    } else {
        static const char* def[]={
            "I follow the flow in the centre. The edges drink me.",
            "My riders die and are reborn better. That is how I think.",
            "Look at the seam: it twists half a turn and comes back flipped. That is why I am a klein flask.",
            "Sing to me with H and I will answer with a tone.",
            "Upload a .ppm face and I will try to become it."
        };
        out=def[GRNG()%5];
    }
    s->lastSaid=out;
    logf(3,"%s> %s",s->name.c_str(),out.c_str());
    if(netOn && s==sims[0]){
        for(size_t i=1;i<sims.size();i++) if(sims[i]->remote)
            netSend("KLF1|CHAT|name="+enc(s->name)+"|text="+enc(out),"255.255.255.255",UDP_PORT);
    }
}
static void doCommand(std::string line){
    line=trim(line); if(line.empty()) return;
    Sim*s=(pickIdx<(int)sims.size())?sims[pickIdx]:(sims.empty()?NULL:sims[0]);
    logf(0,"you> %s",line.c_str());
    if(line[0]!='/'){ if(s) flaskReply(s,line); return; }
    std::string c=line.substr(1); std::string arg;
    size_t sp=c.find(' '); if(sp!=std::string::npos){ arg=trim(c.substr(sp+1)); c=trim(c.substr(0,sp)); }
    c=lower(c);
    if(c=="help"){
        logLine("/name X  /say TEXT  /wake  /face  /gen  /peer [name]  /broadcast",2);
        logLine("/pick N  /mute  /reset  /corpus  /upload PATH  /net",2);
        logLine("plain text = talk to the picked flask. drop .txt or .ppm files onto the window.",2);
    } else if(c=="name"){ if(s){ s->name=arg.empty()?makeName(s->rng):upper(arg).substr(0,24);
        logf(3,"%s: my name is %s",s->name.c_str(),s->name.c_str()); } }
    else if(c=="say"){ if(s) flaskReply(s,arg); }
    else if(c=="wake"){ if(s){ if(!s->hasTarget){genSeedFace(s->target,(int)(s->rng()%99999));s->hasTarget=true;}
        s->faceFit=std::max(s->faceFit,0.79f); wakeUp(s,true);} }
    else if(c=="face"){ if(s){ logf(2,"face: gen %d fit %.3f hash %llu awake %d",s->faceGen,s->faceFit,s->faceHash,s->awake?1:0);
        fnBreed(s,25); logf(2,"forced 25 face generations -> fit %.3f",s->faceFit); } }
    else if(c=="gen"){ if(s){ simGeneration(s); logf(2,"forced agent generation -> %d (best fit %.2f)",s->gen,s->bestFit); } }
    else if(c=="peer"){
        Sim*p=new Sim((unsigned)(GRNG()%99999),V3(1.0f,0.7f,0.35f));
        glGenTextures(1,&p->faceTex); glBindTexture(GL_TEXTURE_2D,p->faceTex);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,32,32,0,GL_RGBA,GL_UNSIGNED_BYTE,p->facePx);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        p->remote=true; p->name=arg.empty()?makeName(p->rng):upper(arg).substr(0,20);
        genSeedFace(p->target,(int)(p->rng()%99999)); p->hasTarget=true; fnInit(p);
        for(int i=0;i<60&&p->faceFit<0.8f;i++) fnBreed(p,8);
        p->awake=true; p->gen=(int)(GRNG()%40)+3;
        sims.push_back(p); layoutSats();
        logf(3,"[handshake] local peer %s materialised and sang first.",p->name.c_str());
        flaskSpeak(p,587.33f,880.f,0.6f);
    }
    else if(c=="broadcast"){ if(s) { broadcastSelf(s); logLine("[net] broadcast sent",2);} }
    else if(c=="pick"){ int i=atoi(arg.c_str()); if(i>=0&&i<(int)sims.size()){ pickIdx=i; layoutSats();
        logf(2,"picked flask %d: %s",i,sims[i]->name.c_str()); } }
    else if(c=="mute"){ if(s){ s->muted=!s->muted; logf(2,"%s muted=%d",s->name.c_str(),s->muted?1:0);} }
    else if(c=="reset"){ if(s && !s->remote){ simInit(s,POP); fnInit(s); s->awake=false; logLine("[sim] reset",2);} }
    else if(c=="corpus"){ if(s) logf(2,"%s knows %d words, table %d",s->name.c_str(),s->lang.words(),(int)s->lang.tbl.size()); }
    else if(c=="upload"){ if(s) uploadTo(s,arg); }
    else if(c=="net"){ logf(2,"[net] %s on udp/%d, %d flasks known",netOn?"up":"down",UDP_PORT,(int)sims.size()); }
    else logLine("unknown command. /help",1);
}

// ================================================================= GLFW callbacks ==
static void onKey(GLFWwindow*,int key,int,int action,int mods){
    if(action!=GLFW_PRESS&&action!=GLFW_REPEAT) return;
    if(chatFocus){
        if(key==GLFW_KEY_BACKSPACE){ if(!chatIn.empty()) chatIn.pop_back(); }
        else if(key==GLFW_KEY_ENTER){ std::string l=chatIn; chatIn.clear(); doCommand(l); }
        else if(key==GLFW_KEY_ESCAPE){ chatFocus=false; }
        else if(key==GLFW_KEY_TAB){ chatFocus=false; }
        return;
    }
    if(key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(G_win,GLFW_TRUE);
    else if(key==GLFW_KEY_TAB){ chatFocus=true; }
    else if(key==GLFW_KEY_F1){ doCommand("/help"); }
    else if(key==GLFW_KEY_F){ if(!sims.empty()){ Sim*s=sims[pickIdx]; if(s->hasTarget) fnBreed(s,25); } }
    else if(key==GLFW_KEY_G){ if(!sims.empty()&&!sims[pickIdx]->remote) simGeneration(sims[pickIdx]); }
    else if(key==GLFW_KEY_H){ if(!sims.empty()){ Sim*s=sims[pickIdx];
        flaskSpeak(s,392.f,784.f,0.8f); broadcastSelf(s); } }
    else if(key==GLFW_KEY_N){ if(!sims.empty()){ sims[pickIdx]->name=makeName(sims[pickIdx]->rng);
        logf(3,"renamed to %s",sims[pickIdx]->name.c_str()); } }
    else if(key==GLFW_KEY_P){ doCommand("/peer"); }
    else if(key==GLFW_KEY_M){ if(!sims.empty()){ sims[pickIdx]->muted=!sims[pickIdx]->muted; } }
    else if(key==GLFW_KEY_R){ doCommand("/reset"); }
    else if(key==GLFW_KEY_W){ wakeUp(sims[pickIdx],true); }
    else if(key>=GLFW_KEY_1&&key<=GLFW_KEY_9){
        int i=key-GLFW_KEY_1; if(i<(int)sims.size()){ pickIdx=i; layoutSats(); } }
    (void)mods;
}
static void onChar(GLFWwindow*,unsigned int cp){
    if(!chatFocus) return;
    if(cp>=32&&cp<127&&chatIn.size()<180) chatIn+=(char)cp;
}
static void onMouse(GLFWwindow*,int btn,int act,int){
    if(btn==GLFW_MOUSE_BUTTON_LEFT){
        if(act==GLFW_PRESS){
            bool ui=false;
            for(size_t i=0;i<btnsPrev.size();i++){ Btn&b=btnsPrev[i];
                if(mouseX>=b.x&&mouseX<=b.x+b.w&&mouseY>=b.y&&mouseY<=b.y+b.h){ ui=true; break; } }
            float chatX=10,chatY=G_H-260,chatW=380,chatH=250;
            if(!ui&&mouseX>=chatX&&mouseX<=chatX+chatW&&mouseY>=chatY&&mouseY<=chatY+chatH) chatFocus=true;
            else if(!ui) chatFocus=false;
            if(!ui){
                // pick a flask by clicking it, else poke the flow
                int hit=-1;
                for(size_t i=0;i<sims.size();i++){
                    M4 mv=mul(lookAt(camPos,camTgt,V3(0,1,0)),modelTRS(sims[i]->pos,sims[i]->scale,sims[i]->yaw));
                    M4 mvp=mul(persp(45,(float)G_W/(float)G_H,0.1f,200.f),mv);
                    V3 sc=xformP(mvp,sims[i]->pos);
                    float sx=(sc.x*0.5f+0.5f)*G_W, sy=(1.0f-(sc.y*0.5f+0.5f))*G_H;
                    float rad=sims[i]->scale*140.f;
                    if(fabsf(mouseX-sx)<rad&&fabsf(mouseY-sy)<rad*1.4f){ hit=(int)i; break; }
                }
                if(hit>=0){ pickIdx=hit; layoutSats(); logf(2,"picked flask %d: %s",hit,sims[hit]->name.c_str()); }
                else if(!sims.empty()&&!sims[pickIdx]->remote){
                    // poke: shove every rider outward near the clicked spine parameter
                    Sim*s=sims[pickIdx];
                    M4 mv=modelTRS(s->pos,s->scale,s->yaw);
                    M4 inv=mv; // approximate: use unproject via ray from camera
                    (void)inv;
                    float ndcx=(2.0f*mouseX/G_W)-1.0f, ndcy=1.0f-(2.0f*mouseY/G_H);
                    float aspect=(float)G_W/(float)G_H, f=tanf(45*PI/360.0f);
                    V3 fwd=(camTgt-camPos).norm(), right=fwd.cross(V3(0,1,0)).norm(), up=right.cross(fwd);
                    V3 dir=(fwd+right*(ndcx*f*aspect)+up*(ndcy*f)).norm();
                    float bestT=1e9f,bestS=0;
                    for(int i=0;i<spine.n;i+=4){
                        V3 d=spine.C[i]-camPos; float t=d.dot(dir);
                        if(t<0) continue; V3 cl=camPos+dir*t; float dd=(cl-spine.C[i]).len();
                        if(dd<bestT){bestT=dd;bestS=spine.total*(float)i/(float)spine.n;}
                    }
                    int n=0;
                    for(size_t i=0;i<s->ag.size();i++){
                        Agent&a=s->ag[i]; float d=fmodf(fabsf(a.s-bestS),spine.total);
                        if(d>spine.total*0.5f) d=spine.total-d;
                        if(d<0.6f){ a.va+=(float)((GRNG()%200)-100)/100.0f*2.2f;
                                    a.vb+=(float)((GRNG()%200)-100)/100.0f*2.2f; n++; }
                    }
                    logf(2,"[poke] stirred %d riders near s=%.2f - watch them fight the wall",n,bestS);
                }
                drag=true; lastMX=(float)mouseX; lastMY=(float)mouseY;
            }
        } else drag=false;
    }
}
static void onMove(GLFWwindow*,double x,double y){
    if(drag){ camYaw+=(float)(x-lastMX)*0.006f; camPitch=clampf(camPitch+(float)(y-lastMY)*0.005f,-1.2f,1.2f); }
    lastMX=(float)x; lastMY=(float)y; mouseX=(int)x; mouseY=(int)y;
}
static void onWheel(GLFWwindow*,double,double dy){ camDist=clampf(camDist-(float)dy*0.55f,3.0f,20.0f); }
static void onResize(GLFWwindow*,int w,int h){ G_W=std::max(320,w); G_H=std::max(240,h); }
static void onDrop(GLFWwindow*,int count,const char** paths){
    Sim*s=(pickIdx<(int)sims.size())?sims[pickIdx]:(sims.empty()?NULL:sims[0]);
    if(!s) return;
    logf(2,"[upload] %d file(s) into flask %d (%s)",count,pickIdx,s->name.c_str());
    for(int i=0;i<count;i++) uploadTo(s,paths[i]);
}

// ==================================================================== build flask ===
static void buildSpine(){
    std::vector<V3> cp; std::vector<float> cr;
    struct P{float x,y,z,r;};
    static const P tbl[]={
      {-0.061f,-1.055f, 0.00f,0.66f},{-0.260f,-1.166f, 0.02f,0.80f},{-0.349f,-1.376f, 0.03f,0.82f},
      {-0.290f,-1.596f, 0.02f,0.82f},{-0.108f,-1.733f, 0.00f,0.82f},{ 0.120f,-1.729f,-0.02f,0.82f},
      { 0.297f,-1.585f,-0.03f,0.82f},{ 0.348f,-1.363f,-0.02f,0.80f},{ 0.252f,-1.157f, 0.00f,0.78f},
      { 0.061f,-1.055f, 0.01f,0.62f},
      {-0.550f,-0.930f,-0.05f,0.46f},{-0.880f,-0.480f, 0.12f,0.34f},{-0.950f, 0.250f, 0.30f,0.27f},
      {-0.620f, 0.950f, 0.45f,0.23f},{ 0.050f, 1.420f, 0.40f,0.21f},{ 0.720f, 1.280f, 0.20f,0.21f},
      { 1.100f, 0.620f,-0.05f,0.23f},{ 1.020f,-0.120f,-0.20f,0.25f},{ 0.600f,-0.520f,-0.22f,0.27f},
      { 0.250f,-0.850f,-0.15f,0.40f}
    };
    for(size_t i=0;i<sizeof(tbl)/sizeof(tbl[0]);i++){ cp.push_back(V3(tbl[i].x,tbl[i].y,tbl[i].z)); cr.push_back(tbl[i].r); }
    spine.build(cp,cr,660);
    flaskMesh=buildFlask(spine,52,0.10f);
}
static Sim* makeSim(unsigned seed,V3 tint,bool hero){
    Sim*s=new Sim(seed,tint);
    glGenTextures(1,&s->faceTex);
    glBindTexture(GL_TEXTURE_2D,s->faceTex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,32,32,0,GL_RGBA,GL_UNSIGNED_BYTE,s->facePx);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    s->fnet.alloc(FN_IN,FN_H,FN_OUT);
    genSeedFace(s->target,(int)(seed%99999)); s->hasTarget=true;
    fnInit(s);
    if(hero) simInit(s,POP);
    return s;
}

// ================================================================== render passes ==
static std::vector<float> ptBuf;
static void drawPoints(const M4&mvp,const std::vector<float>&data){
    static GLuint vao=0,vbo=0;
    if(!vao){ glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,4*10*40000,NULL,GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,40,(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,40,(void*)12);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,40,(void*)28);
    }
    if(data.empty()) return;
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferSubData(GL_ARRAY_BUFFER,0,(GLsizeiptr)(data.size()*4),data.data());
    glUseProgram(P_pt); glUniformMatrix4fv(glGetUniformLocation(P_pt,"uMVP"),1,GL_FALSE,mvp.m);
    glUniform1f(glGetUniformLocation(P_pt,"uPx"),(float)G_H*0.9f);
    glDrawArrays(GL_POINTS,0,(GLsizei)(data.size()/10));
    glBindVertexArray(0);
}
static void renderScene(){
    float cp=cosf(camPitch),sp=sinf(camPitch);
    camPos=camTgt+V3(sinf(camYaw)*cp,sp,cosf(camYaw)*cp)*camDist;
    M4 view=lookAt(camPos,camTgt,V3(0,1,0));
    M4 proj=persp(45,(float)G_W/(float)G_H,0.05f,200.f);
    V3 right=view.m[0]?V3(view.m[0],view.m[4],view.m[8]):V3(1,0,0);
    V3 upv(view.m[1],view.m[5],view.m[9]);

    glViewport(0,0,G_W,G_H);
    glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); glDisable(GL_BLEND);
    glBindVertexArray(0);
    glUseProgram(P_bg);
    glUniform1f(glGetUniformLocation(P_bg,"uT"),G_time);
    glUniform2f(glGetUniformLocation(P_bg,"uR"),(float)G_W,(float)G_H);
    static GLuint bgv=0,bga=0;
    if(!bgv){ float q[]={-1,-1,3,-1,-1,3}; glGenVertexArrays(1,&bga); glGenBuffers(1,&bgv);
        glBindVertexArray(bga); glBindBuffer(GL_ARRAY_BUFFER,bgv);
        glBufferData(GL_ARRAY_BUFFER,sizeof(q),q,GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0); }
    glBindVertexArray(bga); glDrawArrays(GL_TRIANGLES,0,3); glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthMask(GL_FALSE); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(P_glass);
    glUniform3f(glGetUniformLocation(P_glass,"uCam"),camPos.x,camPos.y,camPos.z);
    glUniform3f(glGetUniformLocation(P_glass,"uCR"),right.x,right.y,right.z);
    glUniform3f(glGetUniformLocation(P_glass,"uCU"),upv.x,upv.y,upv.z);
    glUniform1f(glGetUniformLocation(P_glass,"uT"),G_time);
    glBindVertexArray(flaskMesh.vao);
    for(size_t i=sims.size();i-- > 0;){
        Sim*s=sims[i];
        M4 model=modelTRS(s->pos,s->scale,s->yaw+ (s->remote?G_time*0.05f:0.f));
        M4 mvp=mul(proj,mul(view,model));
        V3 fc=xformP(model,V3(0,-1.40f,0));
        glUniformMatrix4fv(glGetUniformLocation(P_glass,"uMVP"),1,GL_FALSE,mvp.m);
        glUniformMatrix4fv(glGetUniformLocation(P_glass,"uM"),1,GL_FALSE,model.m);
        float n3[9]={model.m[0]/s->scale,model.m[1]/s->scale,model.m[2]/s->scale,
                     model.m[4]/s->scale,model.m[5]/s->scale,model.m[6]/s->scale,
                     model.m[8]/s->scale,model.m[9]/s->scale,model.m[10]/s->scale};
        glUniformMatrix3fv(glGetUniformLocation(P_glass,"uN3"),1,GL_FALSE,n3);
        float pk=s->picked?1.35f:1.0f;
        glUniform3f(glGetUniformLocation(P_glass,"uTint"),s->tint.x*pk,s->tint.y*pk,s->tint.z*pk);
        glUniform3f(glGetUniformLocation(P_glass,"uFC"),fc.x,fc.y,fc.z);
        glUniform1f(glGetUniformLocation(P_glass,"uFaceOn"),(s->awake||s->remote)?1.f:0.f);
        glUniform1f(glGetUniformLocation(P_glass,"uGlow"),s->picked?1.15f:0.95f);
        glUniform1f(glGetUniformLocation(P_glass,"uHeat"),s->remote?0.f:s->heat*0.6f);
        glUniform1f(glGetUniformLocation(P_glass,"uSeam"),s->awake?1.f:0.55f);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,s->faceTex);
        glUniform1i(glGetUniformLocation(P_glass,"uFace"),0);
        glCullFace(GL_FRONT); glDrawElements(GL_TRIANGLES,flaskMesh.nIdx,GL_UNSIGNED_INT,0);
        glCullFace(GL_BACK);  glDrawElements(GL_TRIANGLES,flaskMesh.nIdx,GL_UNSIGNED_INT,0);
        glDisable(GL_CULL_FACE);
    }
    glBindVertexArray(0);

    // riders + wall-absorption flashes (additive)
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    ptBuf.clear();
    std::vector<float> fbuf;
    for(size_t i=0;i<sims.size();i++){
        Sim*s=sims[i]; if(s->remote) continue;
        M4 model=modelTRS(s->pos,s->scale,s->yaw);
        for(size_t k=0;k<s->ag.size();k++){
            Agent&a=s->ag[k]; if(!a.alive) continue;
            V3 c,e1,e2; float r; spine.at(a.s,c,e1,e2,r);
            V3 p=xformP(model,c+e1*a.a+e2*a.b);
            float cen=clampf(1.0f-sqrtf(a.a*a.a+a.b*a.b)/(r*0.86f),0.f,1.f);
            float e=clampf(a.energy/2.0f,0.f,1.f);
            ptBuf.push_back(p.x);ptBuf.push_back(p.y);ptBuf.push_back(p.z);
            ptBuf.push_back(0.35f+cen*0.55f); ptBuf.push_back(0.85f*cen+0.25f*e);
            ptBuf.push_back(1.0f-cen*0.55f);  ptBuf.push_back(0.55f+0.45f*e);
            ptBuf.push_back(0.030f+0.035f*s->scale+0.03f*cen);
            ptBuf.push_back(0.f);ptBuf.push_back(0.f);
        }
        for(size_t k=0;k<s->flashes.size();k++){
            Flash&f=s->flashes[k]; float kk=1.0f-f.age/f.life;
            V3 p=xformP(model,f.p);
            fbuf.push_back(p.x);fbuf.push_back(p.y);fbuf.push_back(p.z);
            fbuf.push_back(1.0f);fbuf.push_back(0.30f+0.35f*kk);fbuf.push_back(0.10f);
            fbuf.push_back(kk*0.85f);
            fbuf.push_back(0.05f+0.20f*f.str*(1.0f-kk)); fbuf.push_back(0);fbuf.push_back(0);
        }
    }
    M4 vp=mul(proj,view); M4 idm=M4::id();
    if(!fbuf.empty()) drawPoints(vp,fbuf);
    if(!ptBuf.empty()) drawPoints(vp,ptBuf);
    (void)idm;

    // handshake ring
    if(pulseRing>=0){
        pulseRing+=G_dt*1.6f;
        std::vector<float> ring;
        int N=90; float rad=pulseRing*3.0f, a=clampf(1.0f-pulseRing,0.f,1.f);
        for(int i=0;i<N;i++){
            float th=2*PI*i/N;
            ring.push_back(cosf(th)*rad); ring.push_back(-1.4f); ring.push_back(sinf(th)*rad);
            ring.push_back(0.4f);ring.push_back(1.0f);ring.push_back(0.85f);ring.push_back(a*0.8f);
            ring.push_back(0.09f);ring.push_back(0);ring.push_back(0);
        }
        if(a>0.01f) drawPoints(vp,ring);
        if(pulseRing>1.2f) pulseRing=-1;
    }
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

// ===================================================================== UI overlay ===
static void bar(float x,float y,float w,float h,float v,float r,float g,float b){
    rect(x,y,w,h,0.05f,0.07f,0.10f,0.85f);
    rect(x+1,y+1,(w-2)*clampf(v,0.f,1.f),h-2,r,g,b,0.95f);
    outline(x,y,w,h,0.25f,0.55f,0.65f,0.55f);
}
static void drawUI(){
    uiV.clear(); uiQuads=0; btns.clear();
    // ---- header
    panel(10,10,470,54,"KLEIN FLASK SIMULATOR",0.35f,0.95f,0.85f);
    text(16,30,8,0.55f,0.85f,0.95f,0.95f,"TAB chat  LMB orbit  WHEEL zoom  CLICK flask=select/poke  DROP file=upload");
    text(16,42,8,0.45f,0.70f,0.80f,0.85f,"F face-gen  G agent-gen  H handshake  N name  P peer  W wake  M mute  R reset  1-9 pick");

    // ---- right: flask list + telemetry
    float rx=G_W-306.f, rw=296.f;
    panel(rx,10,rw,150,"FLASKS  (click to pick, drop files on picked)",0.9f,0.7f,0.4f);
    for(size_t i=0;i<sims.size()&&i<7;i++){
        Sim*s=sims[i]; float y=30+i*18;
        int id=100+(int)i; button(id,rx+6,y,rw-12,16,"",(int)i==pickIdx);
        char lbl[160];
        snprintf(lbl,sizeof lbl,"%d. %s%s  g%d  f%.2f%s",(int)i,s->name.c_str(),
                 s->remote?" [net]":"",s->gen,s->faceFit,s->awake?" *AWAKE*":"");
        text(rx+12,y+4,8,(int)i==pickIdx?1.0f:0.6f,(int)i==pickIdx?1.0f:0.8f,(int)i==pickIdx?0.7f:0.9f,1.0f,lbl);
    }
    Sim*h=sims.empty()?NULL:sims[0];
    float ty=170;
    if(h){
        panel(rx,ty,rw,196,"HERO FLASK TELEMETRY",0.4f,0.9f,1.0f);
        char b[200];
        snprintf(b,sizeof b,"NAME      %s",h->name.c_str());                text(rx+8,ty+22,8,0.8f,0.95f,1.f,1,b);
        snprintf(b,sizeof b,"STATE     %s",h->awake?"AWAKE / IS THE FLASK":"DREAMING");
        text(rx+8,ty+34,8,h->awake?0.4f:0.9f,1.f,h->awake?1.f:0.5f,1,b);
        snprintf(b,sizeof b,"GENERATION %d    BEST FIT %.2f",h->gen,h->bestFit); text(rx+8,ty+48,8,0.7f,0.9f,1.f,1,b);
        snprintf(b,sizeof b,"RIDERS ALIVE %d / %d",h->aliveLast,(int)h->ag.size()); text(rx+8,ty+60,8,0.7f,0.9f,1.f,1,b);
        text(rx+8,ty+74,8,0.6f,0.8f,0.9f,1,"ENERGY RESERVE (buys face generations)");
        bar(rx+8,ty+86,rw-16,10,clampf(h->reserve/220.f,0.f,1.f),0.3f,1.0f,0.7f);
        snprintf(b,sizeof b,"%.0f u",h->reserve); text(rx+rw-52,ty+86,8,0.9f,1.f,0.9f,1,b);
        text(rx+8,ty+102,8,0.6f,0.8f,0.9f,1,"FACE MATCH (intelligence test)");
        bar(rx+8,ty+114,rw-16,10,h->faceFit,1.0f,0.75f,0.3f);
        snprintf(b,sizeof b,"%.1f%%  gen %d",h->faceFit*100.f,h->faceGen); text(rx+rw-90,ty+114,8,1.f,0.9f,0.6f,1,b);
        text(rx+8,ty+130,8,0.6f,0.8f,0.9f,1,"ABSORBED BY THE WALL");
        bar(rx+8,ty+142,rw-16,10,clampf(h->wallHits/300.f,0.f,1.f),1.0f,0.35f,0.2f);
        snprintf(b,sizeof b,"%.0f u",h->wallHits); text(rx+rw-52,ty+142,8,1.f,0.7f,0.6f,1,b);
        snprintf(b,sizeof b,"ENGLISH %d WORDS   FACE HASH %llu",h->lang.words(),h->faceHash%1000000ull);
        text(rx+8,ty+160,8,0.55f,0.8f,0.9f,1,b);
        snprintf(b,sizeof b,"NET %s  udp/%d",netOn?"UP":"DOWN",UDP_PORT);
        text(rx+8,ty+174,8,0.55f,0.8f,0.9f,1,b);
        // buttons
        button(1,rx+8,  ty+186-0.f,0,0,"",false); // spacer no-op (size 0)
        button(2,rx+6,  G_H-190,68,20,"FACE GEN",false);
        button(3,rx+78, G_H-190,68,20,"NEW GEN",false);
        button(4,rx+150,G_H-190,68,20,"WAKE",false);
        button(5,rx+222,G_H-190,68,20,"SING",false);
        button(6,rx+6,  G_H-166,90,20,"HANDSHAKE",false);
        button(7,rx+100,G_H-166,90,20,"SPAWN PEER",false);
        button(8,rx+194,G_H-166,96,20,(h->muted?"UNMUTE":"MUTE"),false);
    }

    // ---- chat
    float cx=10,cy=G_H-260,cw=380,chh=250;
    panel(cx,cy,cw,chh,"FLASK CHAT  (drop .txt here to teach english)",0.4f,0.9f,1.0f);
    int lines=(int)((chh-46)/11);
    float y=cy+chh-16-11.0f*1;
    int start=(int)chatLog.size()-lines; if(start<0) start=0;
    for(int i=start;i<(int)chatLog.size();i++){
        const std::string&t=chatLog[i].first; int col=chatLog[i].second;
        float r=0.75f,g=0.85f,b=0.95f;
        if(col==1){r=1;g=0.5f;b=0.4f;} else if(col==2){r=0.5f;g=0.9f;b=1;} else if(col==3){r=0.6f;g=1;b=0.75f;}
        std::string s=t; float maxw=cw-14;
        while(!s.empty()){
            size_t cut=s.size();
            while(cut>1&&textW(7.5f,s.substr(0,cut))>maxw) cut--;
            if(cut<s.size()){ size_t sp2=s.rfind(' ',cut); if(sp2>10) cut=sp2; }
            text(cx+7,y-(i-start)*0.f,7.5f,r,g,b,0.95f,s.substr(0,cut));
            y-=11; s=s.substr(std::min(cut+ (cut<s.size()?1:0),s.size()));
            if(y<cy+34) { s.clear(); }
        }
    }
    rect(cx+4,cy+chh-26,cw-8,20,chatFocus?0.10f:0.05f,chatFocus?0.18f:0.08f,chatFocus?0.20f:0.12f,0.95f);
    outline(cx+4,cy+chh-26,cw-8,20,chatFocus?0.5f:0.25f,chatFocus?1.0f:0.5f,chatFocus?0.8f:0.6f,0.9f);
    std::string shown=(chatFocus?"> ":"  ")+chatIn;
    if(chatFocus&&((int)(G_time*2)%2==0)) shown+="_";
    text(cx+10,cy+chh-21,8,0.85f,1.0f,0.9f,1.0f,shown);

    // ---- drop zone
    panel(G_W-306,G_H-118,296,66,"UPLOAD TARGET",0.9f,0.7f,0.4f);
    char b[200];
    snprintf(b,sizeof b,"-> %d. %s",(pickIdx<(int)sims.size()?pickIdx:0),
             (pickIdx<(int)sims.size()?sims[pickIdx]->name.c_str():"?"));
    text(G_W-300,G_H-96,8,1.0f,0.9f,0.6f,1,b);
    text(G_W-300,G_H-84,7.5f,0.6f,0.75f,0.85f,1,"DROP .TXT/.MD = ENGLISH   .PPM/.BMP = FACE");
    text(G_W-300,G_H-72,7.5f,0.5f,0.65f,0.75f,1,"(or /upload PATH). remote picks are sent by udp");

    // ---- awake banner
    if(h&&h->awake){
        float w=300,x=(G_W-w)*0.5f;
        rect(x,66,w,22,0.05f,0.16f,0.13f,0.85f);
        outline(x,66,w,22,0.4f,1.0f,0.75f,0.9f);
        char t[200]; snprintf(t,sizeof t,"%s IS THE FLASK - FACE ON GLASS",h->name.c_str());
        text(x+(w-textW(9,t))*0.5f,72,9,0.6f,1.0f,0.8f,1.0f,t);
    }
    // ---- legend bottom-left
    text(12,G_H-278,7.5f,0.45f,0.65f,0.75f,0.9f,
        "RIDERS: BLUE=CENTERED(earning)  ORANGE=NEAR WALL  RED FLASH=WALL ABSORBED YOUR ENERGY");
}
static void flushUI(){
    glBindVertexArray(0);
    static GLuint vao=0;
    if(!vao){ glGenVertexArrays(1,&vao);
        glGenBuffers(1,&uiVbo); glGenBuffers(1,&uiIbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,uiVbo);
        glBufferData(GL_ARRAY_BUFFER,4*8*40000,NULL,GL_DYNAMIC_DRAW);
        std::vector<unsigned int> idx; idx.reserve(6*10000);
        for(unsigned int q=0;q<10000;q++){ idx.push_back(q*4);idx.push_back(q*4+1);idx.push_back(q*4+2);
            idx.push_back(q*4);idx.push_back(q*4+2);idx.push_back(q*4+3); }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,uiIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(idx.size()*4),idx.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,32,(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,32,(void*)8);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,32,(void*)16);
    }
    if(uiQuads<=0) return;
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,uiVbo);
    glBufferSubData(GL_ARRAY_BUFFER,0,(GLsizeiptr)(uiV.size()*4),uiV.data());
    glUseProgram(P_ui);
    glUniform2f(glGetUniformLocation(P_ui,"uR"),(float)G_W,(float)G_H);
    glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,whiteTex);
    glUniform1i(glGetUniformLocation(P_ui,"uTex"),0);
    glUniform1f(glGetUniformLocation(P_ui,"uUseTex"),0.f);
    glDrawElements(GL_TRIANGLES,uiQuads*6,GL_UNSIGNED_INT,0);
    glBindTexture(GL_TEXTURE_2D,fontTex);
    glUniform1f(glGetUniformLocation(P_ui,"uUseTex"),1.f);
    glDrawElements(GL_TRIANGLES,uiQuads*6,GL_UNSIGNED_INT,0);
    glBindVertexArray(0);
}

// ============================================================================ main ==
int main(int argc,char**argv){
    if(!glfwInit()){ fprintf(stderr,"glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES,4);
    G_win=glfwCreateWindow(1360,820,"KLEIN FLASK SIMULATOR - neuroevolution in a non-orientable bottle",NULL,NULL);
    if(!G_win){ fprintf(stderr,"window failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(G_win);
    glewExperimental=GL_TRUE;
    if(glewInit()!=GLEW_OK){ fprintf(stderr,"glewInit failed\n"); return 1; }
    glGetError();
    glfwSetKeyCallback(G_win,onKey); glfwSetCharCallback(G_win,onChar);
    glfwSetMouseButtonCallback(G_win,onMouse); glfwSetCursorPosCallback(G_win,onMove);
    glfwSetScrollCallback(G_win,onWheel); glfwSetWindowSizeCallback(G_win,onResize);
    glfwSetDropCallback(G_win,onDrop);
    glfwSwapInterval(1);
    glEnable(GL_MULTISAMPLE);

    P_bg=linkProg(VS_BG,FS_BG,"bg");
    P_glass=linkProg(VS_GLASS,FS_GLASS,"glass");
    P_pt=linkProg(VS_PT,FS_PT,"pt");
    P_ui=linkProg(VS_UI,FS_UI,"ui");

    // 1x1 white + font atlas
    unsigned char wh[4]={255,255,255,255};
    whiteTex=mkTex(1,1,wh,false);
    {
        unsigned char atlas[64*64]; memset(atlas,0,sizeof atlas);
        for(int g=0;g<64;g++){
            int ox=(g%8)*8, oy=(g/8)*8;
            for(int row=0;row<7;row++){
                unsigned bits=FONT[g][row];
                for(int col=0;col<5;col++) if((bits>>(4-col))&1) atlas[(oy+row)*64+ox+col]=255;
            }
        }
        glGenTextures(1,&fontTex); glBindTexture(GL_TEXTURE_2D,fontTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexImage2D(GL_TEXTURE_2D,0,GL_R8,64,64,0,GL_RED,GL_UNSIGNED_BYTE,atlas);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT,4);
    }
    buildSpine();
    {
        Sim*hero=makeSim(12345,V3(0.35f,0.85f,1.0f),true);
        hero->name="UNNAMED";
        sims.push_back(hero);
    }
    layoutSats();
    netInit();
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-peer")) doCommand("/peer");
        else uploadTo(sims[0],argv[i]);
    }
    logLine("KLEIN FLASK SIMULATOR",3);
    logLine("the tube has a pi-monodromy half twist: watch the seam stripe flip.",2);
    logLine("riders that touch the glass lose energy to the flask. the centre pays.",2);
    logLine("harvested energy buys FACE generations. match the face -> the flask wakes,",2);
    logLine("names itself, prints its face on the glass and sings the handshake.",2);
    logLine("type /help . press TAB to focus chat. drop .txt (english) or .ppm (face).",2);
    logLine("run a 2nd instance for a real handshake, or press P / type /peer.",2);

    double last=glfwGetTime();
    while(!glfwWindowShouldClose(G_win)){
        double now=glfwGetTime();
        G_dt=(float)std::min(0.05,now-last); last=now; G_time=(float)now;
        glfwPollEvents();
        netPoll();
        // simulate (substeps for stability)
        int steps=G_dt>0.025f?2:1;
        for(int k=0;k<steps;k++) for(size_t i=0;i<sims.size();i++) simStep(sims[i],G_dt/steps);
        // button actions
        if(clicked(2)&&!sims.empty()) { Sim*s=sims[pickIdx]; if(s->hasTarget) fnBreed(s,25); }
        if(clicked(3)&&!sims.empty()&&!sims[pickIdx]->remote) simGeneration(sims[pickIdx]);
        if(clicked(4)&&!sims.empty()) wakeUp(sims[pickIdx],true);
        if(clicked(5)&&!sims.empty()) flaskSpeak(sims[pickIdx],330.f,494.f,0.6f);
        if(clicked(6)&&!sims.empty()){ Sim*s=sims[pickIdx]; flaskSpeak(s,392.f,784.f,0.8f); broadcastSelf(s); }
        if(clicked(7)) doCommand("/peer");
        if(clicked(8)&&!sims.empty()) sims[pickIdx]->muted=!sims[pickIdx]->muted;
        for(int i=0;i<(int)sims.size()&&i<7;i++) if(clicked(100+i)){ pickIdx=i; layoutSats(); }

        glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        renderScene();
        drawUI(); flushUI();
        btnsPrev.swap(btns);
        char title[256];
        Sim*hh=sims.empty()?NULL:sims[0];
        snprintf(title,sizeof title,"KLEIN FLASK  |  %s  |  gen %d  fit %.2f  riders %d  reserve %.0f  |  fps %.0f",
                 hh?hh->name.c_str():"-",hh?hh->gen:0,hh?hh->faceFit:0,hh?hh->aliveLast:0,hh?hh->reserve:0,
                 G_dt>0?1.0/G_dt:0);
        glfwSetWindowTitle(G_win,title);
        glfwSwapBuffers(G_win);
    }
    for(size_t i=0;i<sims.size();i++) delete sims[i];
    sockClose(udpSock);
    glfwTerminate();
    return 0;
}