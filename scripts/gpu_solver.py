"""Experimental FP64 CUDA solver; never substitutes for frozen CPU results.

Uses NVIDIA cuda-bindings/NVRTC and PyTorch-owned device buffers. Supports only
F8's 300-shell implicit Yukawa-tu/index-5 configuration. No dependency changes
or edits to gravop.solver are needed. Numerical gates precede any adoption.
"""
import ctypes
from pathlib import Path
import sys
import time
import numpy as np
import torch
from cuda.bindings import driver, nvrtc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gravop import solver

SOURCE=Path(__file__).with_name('cuda')/'halo.cu'


def checked(value):
    if int(value[0]): raise RuntimeError(str(value[0]))
    return value[1] if len(value)==2 else value[1:]


class GPUSolver:
    def __init__(self, device=0):
        self.device=device
        torch.cuda.set_device(device)
        torch.empty(1,device=f'cuda:{device}')  # retain the primary context
        source=SOURCE.read_text()
        # Exact host constants from the frozen CPU reference.
        for name in ('A_COEF','B_COEF'):
            lines=source.splitlines()
            source='\n'.join(f'#define {name} {float(getattr(solver,name))!r}' if l.startswith(f'#define {name} ') else l for l in lines)
        program=checked(nvrtc.nvrtcCreateProgram(source.encode(),b'halo.cu',0,[],[]))
        options=[b'--gpu-architecture=compute_89',b'--fmad=false',b'--std=c++17']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        size=checked(nvrtc.nvrtcGetProgramLogSize(program));log=bytearray(size)
        checked(nvrtc.nvrtcGetProgramLog(program,log));self.compile_log=log.decode().rstrip('\0')
        if int(result[0]): raise RuntimeError(self.compile_log)
        size=checked(nvrtc.nvrtcGetPTXSize(program));ptx=bytearray(size)
        checked(nvrtc.nvrtcGetPTX(program,ptx));checked(nvrtc.nvrtcDestroyProgram(program))
        self.module=checked(driver.cuModuleLoadData(ptx))
        self.evolve=checked(driver.cuModuleGetFunction(self.module,b'evolve'))
        self.hydro=checked(driver.cuModuleGetFunction(self.module,b'hydro_probe'))
        self.initial=np.stack(solver.nfw_profile(n_shells=300))
        self.initial_gpu=self.tensor(self.initial)

    def tensor(self, array):
        return torch.as_tensor(np.ascontiguousarray(array),device=f'cuda:{self.device}')

    def launch(self, function, grid, args, threads=512):
        torch.cuda.set_device(self.device)
        values=[ctypes.c_void_p(v.data_ptr()) if isinstance(v,torch.Tensor) else
                ctypes.c_int(v) if isinstance(v,int) else ctypes.c_double(v) for v in args]
        pointers=(ctypes.c_void_p*len(values))(*[ctypes.addressof(v) for v in values])
        stream=driver.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        checked(driver.cuLaunchKernel(function,grid,1,1,threads,1,1,0,stream,ctypes.addressof(pointers),0))

    def probe_hydro(self,initial=None):
        initial=self.initial if initial is None else initial
        initial_gpu=self.tensor(initial)
        out=self.tensor(np.zeros((3,300)));err=self.tensor(np.zeros(1))
        self.launch(self.hydro,1,[initial_gpu,out,err]);torch.cuda.synchronize(self.device)
        return out.cpu().numpy(),float(err.cpu()[0])

    def run(self, log_params, level=1, t_end=900., max_steps=200_000_000,threads=512):
        params=np.asarray(log_params,dtype=np.float64)
        if params.ndim!=2 or params.shape[1]!=2 or not np.isfinite(params).all():raise ValueError('finite (batch,2) log parameters required')
        if level not in (1,2):raise ValueError('level must be 1 or 2')
        if threads!=512:raise ValueError('PCR prototype requires exactly 512 threads per halo')
        if max_steps<1 or t_end<=0:raise ValueError('positive step cap and end time required')
        if len(params)==0:raise ValueError('empty batch')
        from e9_inference import R_OBS
        n=len(params);gpu_params=self.tensor(10.**params)
        times=self.tensor(np.array([100.,400.,900.]));radii=self.tensor(R_OBS)
        profiles=self.tensor(np.full((n,3,20),np.nan));valid=self.tensor(np.zeros((n,3),np.int32))
        status=self.tensor(np.zeros(n,np.int32));final_t=self.tensor(np.zeros(n));steps=self.tensor(np.zeros(n,np.int32));state=self.tensor(np.zeros((n,3,300)))
        eps_t,snap_rho,snap_t=(1e-4,.01,.025) if level==1 else (3e-5,.005,.0125)
        torch.cuda.synchronize(self.device);start=time.perf_counter()
        self.launch(self.evolve,n,[self.initial_gpu,gpu_params,times,radii,profiles,valid,status,final_t,steps,state,n,
                     eps_t,1e-14,float(t_end),int(max_steps),snap_rho,snap_t,6000],threads)
        torch.cuda.synchronize(self.device);seconds=time.perf_counter()-start
        return dict(params=params,profiles=profiles.cpu().numpy(),valid=valid.cpu().numpy().astype(bool),
                    status=status.cpu().numpy(),t_end=final_t.cpu().numpy(),steps=steps.cpu().numpy(),
                    final_state=state.cpu().numpy(),seconds=seconds,device=self.device)
