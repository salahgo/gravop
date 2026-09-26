// Experimental batched port of gravop/solver.py; GPLv3, same physical scheme.
// One block per independent halo. FP64, no fast math, no fused multiply-add.
#define N 300
#define A_COEF 2.256758334191025
#define B_COEF 1.384729571019934
#define C_COEF 0.753
#define GAMMA (5.0/3.0)
struct Work {
 double r[N], rho[N], p[N], u[N], L[N], kn[N], K[N], coef[N];
 double lo[N], di[N], up[N], rhs[N], unew[N], dr[N];
 double prev_r[N], prev_rho[N], re[N+1], r2[N+1], r3[N+1];
 double t, dt, dtprev, last_t, last_ln, rho0, maxrel;
 int status, steps, snaps, accept, want, done[3];
};
__device__ double factor(double s) {
 double s4=s*s*s*s+1e-8;
 double g=1.5*log(1.0+pow(s4*0.251115,0.41))/(24.0*0.41*s4);
 return pow(1.0+1.0/pow(g,0.71),-1.0/0.71);
}
__device__ double mass(const double *m,int j) { return j==0?0.0:m[j-1]; }
__device__ void derived(Work &w,double sig,double vel) {
 for(int i=threadIdx.x;i<N;i+=blockDim.x) {
  w.u[i]=1.5*w.p[i]/w.rho[i];
  double v=sqrt(w.p[i]/w.rho[i]),F=factor(v/vel);
  double ks=sig*F/(B_COEF*v),kl=1.0/(A_COEF*C_COEF*v*w.p[i]*sig*F);
  w.K[i]=1.0/(ks+kl);w.kn[i]=1.0/(sqrt(w.p[i])*sig*F);
 }
 __syncthreads();
 for(int i=threadIdx.x;i<N;i+=blockDim.x) {
  if(i==N-1){w.L[i]=0.0;w.coef[i]=0.0;}
  else {
   double gap=i==0?w.r[1]*0.5:(w.r[i+1]-w.r[i-1])*0.5;
   w.coef[i]=w.r[i]*w.r[i]*0.5*(w.K[i]+w.K[i+1])/gap;
   // Match the reference expression ordering for luminosity.
   w.L[i]=-w.r[i]*w.r[i]*0.5*(w.K[i]+w.K[i+1])*(w.u[i+1]-w.u[i])/gap;
  }
 }
 __syncthreads();
}
// Parallel cyclic reduction, one thread per row. All old coefficients
// are read before any writes. The halo equations and time-step rules are unchanged.
__device__ void pcr(Work &w,int n) {
 int i=threadIdx.x;
 for(int stride=1;stride<n;stride*=2){
  double a=0.0,b=0.0,c=0.0,d=0.0;
  if(i<n){
   b=w.di[i];d=w.rhs[i];
   if(i>=stride){double pivot=w.di[i-stride];if(pivot==0.0)pivot=1e-300;double v=w.lo[i]/pivot;
    b-=v*w.up[i-stride];d-=v*w.rhs[i-stride];a=-v*w.lo[i-stride];}
   if(i+stride<n){double pivot=w.di[i+stride];if(pivot==0.0)pivot=1e-300;double v=w.up[i]/pivot;
    b-=v*w.lo[i+stride];d-=v*w.rhs[i+stride];c=-v*w.up[i+stride];}
  }
  __syncthreads();
  if(i<n){w.lo[i]=a;w.di[i]=b;w.up[i]=c;w.rhs[i]=d;}
  __syncthreads();
 }
}
__device__ void conduct_try(Work &w,const double *m) {
 for(int i=threadIdx.x;i<N;i+=blockDim.x) {
  double dm=m[i]-(i>0?m[i-1]:0.0),c=w.dt/dm,am=i>0?w.coef[i-1]:0.0;
  w.di[i]=1.0+c*(w.coef[i]+am);w.up[i]=-c*w.coef[i];w.lo[i]=-c*am;w.rhs[i]=w.u[i];
 }
 __syncthreads();
 pcr(w,N);
 if(threadIdx.x<N)w.unew[threadIdx.x]=w.rhs[threadIdx.x]/w.di[threadIdx.x];
 __syncthreads();
 if(threadIdx.x==0) {
  double mx=0.0;
  for(int i=0;i<N;++i){double v=fabs(w.unew[i]-w.u[i])/w.u[i];if(v>mx)mx=v;}
  w.maxrel=mx;
 }
 __syncthreads();
}
__device__ void hydro(Work &w,const double *m) {
 for(int i=threadIdx.x;i<N+1;i+=blockDim.x){w.re[i]=i==0?0.0:w.r[i-1];w.r2[i]=w.re[i]*w.re[i];w.r3[i]=w.r2[i]*w.re[i];}
 __syncthreads();
 for(int i=threadIdx.x;i<N-1;i+=blockDim.x){
  if(i<N-2){
   w.lo[i+1]=((12.*GAMMA*w.p[i+1]*w.r2[i+1]*w.r2[i+2]
    +mass(m,i+2)*(-3.*w.r2[i+1]*w.re[i+3]*w.rho[i+1]
     +w.r3[i+1]*(2.*w.rho[i+1]-w.rho[i+2])+w.r3[i+2]*(w.rho[i+1]+w.rho[i+2])))
    /(4.*(w.r3[i+1]-w.r3[i+2])));
   w.up[i]=((12.*GAMMA*w.p[i+1]*w.r2[i+1]*w.r2[i+2]
    +mass(m,i+1)*(w.r3[i+1]*(w.rho[i]+w.rho[i+1])
     -w.r2[i+2]*(w.re[i+2]*w.rho[i]+3.*w.re[i]*w.rho[i+1]-2.*w.re[i+2]*w.rho[i+1])))
    /(4.*(w.r3[i+1]-w.r3[i+2])));
  }
  w.di[i]=(w.re[i+1]*(-4.*w.p[i]*(2.*w.r3[i]+(-2.+3.*GAMMA)*w.r3[i+1])*(w.r3[i+1]-w.r3[i+2])
   -4.*w.p[i+1]*(w.r3[i]-w.r3[i+1])*((-2.+3.*GAMMA)*w.r3[i+1]+2.*w.r3[i+2])
   +3.*mass(m,i+1)*w.re[i+1]*(w.re[i]-w.re[i+2])*(w.r3[i+2]*w.rho[i]+w.r3[i]*w.rho[i+1]-w.r3[i+1]*(w.rho[i]+w.rho[i+1])))
   /(4.*(w.r3[i]-w.r3[i+1])*(w.r3[i+1]-w.r3[i+2])));
  w.rhs[i]=-(-w.p[i]*w.r2[i+1]+w.p[i+1]*w.r2[i+1]
   +0.25*mass(m,i+1)*(w.re[i+2]-w.re[i])*(w.rho[i]+w.rho[i+1]));
 }
 if(threadIdx.x==0){w.lo[0]=0.0;w.up[N-2]=0.0;}
 __syncthreads();
 pcr(w,N-1);
 if(threadIdx.x<N-1){double piv=w.di[threadIdx.x];if(piv==0.0)piv=1e-300;w.dr[threadIdx.x]=w.rhs[threadIdx.x]/piv;}
 __syncthreads();
 if(threadIdx.x==0){
  w.dr[N-1]=0.0;double alpha=1.0,mx=0.0;
  for(int i=0;i<N-1;++i){
   double v=fabs(w.dr[i]/w.r[i]);if(v>mx)mx=v;
   double gi=w.r[i]-(i>0?w.r[i-1]:0.0),go=w.r[i+1]-w.r[i],gap=gi<go?gi:go,adr=fabs(w.dr[i]);
   if(adr>0.25*gap){double cand=0.25*gap/adr;if(cand<alpha)alpha=cand;}
  }
  for(int i=0;i<N;++i){
   double ri=i<N-1?w.r[i]*w.r[i]*w.dr[i]:0.0,rim=i>0?w.r[i-1]*w.r[i-1]*w.dr[i-1]:0.0;
   double vol=(w.r[i]*w.r[i]*w.r[i]-(i>0?w.r[i-1]*w.r[i-1]*w.r[i-1]:0.0))/3.;
   double rel=GAMMA*fabs(ri-rim)/vol;
   if(rel>0.25){double cand=0.25/rel;if(cand<alpha)alpha=cand;}
  }
  if(alpha<1.0)for(int i=0;i<N-1;++i)w.dr[i]*=alpha;
  w.maxrel=mx;
 }
 __syncthreads();
 for(int i=threadIdx.x;i<N;i+=blockDim.x){
  double ri=w.r[i]*w.r[i]*w.dr[i],rim=i>0?w.r[i-1]*w.r[i-1]*w.dr[i-1]:0.0;
  double vol=(w.r[i]*w.r[i]*w.r[i]-(i>0?w.r[i-1]*w.r[i-1]*w.r[i-1]:0.0))/3.;
  w.rho[i]+=-w.rho[i]*(ri-rim)/vol;w.p[i]+=-GAMMA*w.p[i]*(ri-rim)/vol;
 }
 __syncthreads();
 for(int i=threadIdx.x;i<N-1;i+=blockDim.x)w.r[i]+=w.dr[i];
 __syncthreads();
}
__device__ double interp(const double *r,const double *rho,double target){
 double first=r[0]*0.5,last=(r[N-2]+r[N-1])*0.5;
 if(target<=first)return log10(rho[0]);if(target>=last)return log10(rho[N-1]);
 int low=0,high=N-1;
 while(high-low>1){int mid=(low+high)/2;double rm=mid==0?r[0]*0.5:(r[mid-1]+r[mid])*0.5;if(rm<=target)low=mid;else high=mid;}
 double x0=log10(low==0?r[0]*0.5:(r[low-1]+r[low])*0.5),x1=log10((r[high-1]+r[high])*0.5);
 double y0=log10(rho[low]),y1=log10(rho[high]);return y0+(y1-y0)/(x1-x0)*(log10(target)-x0);
}
__device__ void snapshot(Work &w,const double *times,const double *radii,double *profiles,int *valid){
 for(int j=0;j<3;++j){
  if(!w.done[j] && times[j]<=w.t+1e-9){
   double a=(times[j]-w.last_t)/(w.t-w.last_t);a=fmin(1.0,fmax(0.0,a));
   for(int k=threadIdx.x;k<20;k+=blockDim.x)profiles[j*20+k]=(1.0-a)*interp(w.prev_r,w.prev_rho,radii[k])+a*interp(w.r,w.rho,radii[k]);
   if(threadIdx.x==0){valid[j]=1;}
  }
 }
 __syncthreads();
 if(threadIdx.x==0)for(int j=0;j<3;++j)if(times[j]<=w.t+1e-9)w.done[j]=1;
 for(int i=threadIdx.x;i<N;i+=blockDim.x){w.prev_r[i]=w.r[i];w.prev_rho[i]=w.rho[i];}
 __syncthreads();
 if(threadIdx.x==0){w.last_t=w.t;w.last_ln=log(w.rho[3]);++w.snaps;}
 __syncthreads();
}
extern "C" __global__ void evolve(const double *initial,const double *parameters,const double *times,const double *radii,
 double *profiles,int *valid,int *status,double *final_t,int *steps_out,double *final_state,
 int count,double eps_t,double eps_r,double t_end,int max_steps,double snap_dlnrho,double snap_dlnt,int max_snaps){
 int h=blockIdx.x;if(h>=count)return;__shared__ Work w;
 const double *m=initial+N;double sig=parameters[h*2],vel=parameters[h*2+1];
 for(int i=threadIdx.x;i<N;i+=blockDim.x){w.r[i]=initial[i];w.rho[i]=initial[2*N+i];w.p[i]=initial[3*N+i];w.prev_r[i]=w.r[i];w.prev_rho[i]=w.rho[i];}
 if(threadIdx.x==0){w.t=0.0;w.dtprev=-1.0;w.last_t=0.0;w.rho0=initial[2*N+3];w.last_ln=log(w.rho0);w.status=3;w.steps=0;w.snaps=1;for(int j=0;j<3;++j){w.done[j]=0;valid[h*3+j]=0;}}
 __syncthreads();derived(w,sig,vel);
 while(w.steps<max_steps){
  if(threadIdx.x==0){
   ++w.steps;double dt1=1e300,dt2=1e300;
   for(int i=0;i<N;++i){double dm=m[i]-(i>0?m[i-1]:0.0),dl=(w.L[i]-(i>0?w.L[i-1]:0.0))/dm;
    if(dl!=0.0){double cand=fabs(w.u[i]/dl);if(cand<dt1)dt1=cand;}
    double cand=1e300/(w.rho[i]*sqrt(w.p[i]/w.rho[i]));if(cand<dt2)dt2=cand;
   }
   w.dt=eps_t*(dt1<dt2?dt1:dt2);if(w.dtprev>0.0 && w.dt>1.5*w.dtprev)w.dt=1.5*w.dtprev;
   if(!isfinite(w.dt)||w.dt<=0.0)w.status=4;w.accept=0;
  }
  __syncthreads();if(w.status==4)break;
  for(int attempt=0;attempt<25;++attempt){
   conduct_try(w,m);
   if(threadIdx.x==0){if(w.maxrel<=5.0*eps_t)w.accept=1;else w.dt*=0.5;}
   __syncthreads();if(w.accept)break;
  }
  if(!w.accept){if(threadIdx.x==0)w.status=4;__syncthreads();break;}
  for(int i=threadIdx.x;i<N;i+=blockDim.x){double du=w.unew[i]-w.u[i];w.p[i]+=w.p[i]*du/w.u[i];w.u[i]=w.unew[i];}
  if(threadIdx.x==0){w.dtprev=w.dt;w.t+=w.dt;}
  __syncthreads();
  for(int nh=0;nh<50;++nh){hydro(w,m);if(w.maxrel<eps_r)break;}
  derived(w,sig,vel);
  if(threadIdx.x==0){
   for(int i=0;i<N;++i)if(!(w.rho[i]>0.0 && w.p[i]>0.0 && isfinite(w.rho[i]) && isfinite(w.p[i])))w.status=4;
   w.want=(fabs(log(w.rho[3])-w.last_ln)>=snap_dlnrho)||(w.last_t>0.0 && (w.t-w.last_t)/w.last_t>=snap_dlnt)||(w.last_t==0.0 && w.t>0.0);
  }
  __syncthreads();if(w.status==4)break;
  if(w.want && w.snaps<max_snaps-1)snapshot(w,times,radii,profiles+h*60,valid+h*3);
  if(threadIdx.x==0){
   if(w.rho[3]>=1e3*w.rho0)w.status=0;
   else if(w.t>=t_end)w.status=1;
   else {double km=w.kn[0];for(int i=1;i<N;++i)if(w.kn[i]<km)km=w.kn[i];if(km<=0.02)w.status=2;}
  }
  __syncthreads();if(w.status!=3)break;
 }
 snapshot(w,times,radii,profiles+h*60,valid+h*3);
 if(threadIdx.x==0){status[h]=w.status;final_t[h]=w.t;steps_out[h]=w.steps;}
 for(int i=threadIdx.x;i<N;i+=blockDim.x){final_state[h*3*N+i]=w.r[i];final_state[h*3*N+N+i]=w.rho[i];final_state[h*3*N+2*N+i]=w.p[i];}
}
extern "C" __global__ void hydro_probe(const double *initial,double *out,double *error){
 __shared__ Work w;const double *m=initial+N;
 for(int i=threadIdx.x;i<N;i+=blockDim.x){w.r[i]=initial[i];w.rho[i]=initial[2*N+i];w.p[i]=initial[3*N+i];}
 __syncthreads();hydro(w,m);
 for(int i=threadIdx.x;i<N;i+=blockDim.x){out[i]=w.r[i];out[N+i]=w.rho[i];out[2*N+i]=w.p[i];}
 if(threadIdx.x==0)error[0]=w.maxrel;
}
