"""Positive adaptive dyadic trapezoidal quadrature on a two-dimensional box.

Embedded differences estimate integration error; they do not prove convergence
for unsampled narrow features. Callers must supply an initial global survey and
check successive rules. Coordinates are integers to make node reuse exact.
"""
import numpy as np


class DyadicMesh:
    def __init__(self, base=32, lattice=4096, cells=None):
        if base<1 or lattice%base or lattice//base%2:
            raise ValueError('base cells must admit integer midpoint nodes')
        self.lattice=int(lattice)
        step=lattice//base
        self.cells=(np.array([(i*step,j*step,step) for i in range(base) for j in range(base)],dtype=int)
                    if cells is None else np.asarray(cells,dtype=int))

    def stencil(self):
        c=self.cells; nodes=[]
        for i in range(3):
            for j in range(3):nodes.append(c[:,:2]+c[:,2,None]*np.array([i,j])//2)
        nodes=np.stack(nodes,1)
        unique,inverse=np.unique(nodes.reshape(-1,2),axis=0,return_inverse=True)
        return unique,inverse.reshape(len(c),9)

    def assess(self, values, indices):
        """Fine integrals and conservative raw embedded differences per output."""
        values=np.asarray(values,dtype=float)
        if values.ndim==1:values=values[:,None]
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('integrands must be finite and nonnegative')
        area=(self.cells[:,2]/self.lattice)**2
        fine=np.zeros((len(self.cells),values.shape[1]));coarse=np.zeros_like(fine)
        weights=np.outer([1.,2.,1.],[1.,2.,1.]).ravel()/16
        for k,w in enumerate(weights):fine+=w*values[indices[:,k]]
        for k in (0,2,6,8):coarse+=.25*values[indices[:,k]]
        fine*=area[:,None];coarse*=area[:,None]
        error=np.abs(fine-coarse); total=fine.sum(0)
        if (total<=0).any():raise ValueError('integrand has no sampled positive mass')
        relative=error.sum(0)/total
        priority=(error/total).max(1)
        return dict(integral=total,relative_error=relative,priority=priority)

    def weights(self, indices, n_nodes):
        area=(self.cells[:,2]/self.lattice)**2
        out=np.zeros(n_nodes);w=np.outer([1.,2.,1.],[1.,2.,1.]).ravel()/16
        for k in range(9):np.add.at(out,indices[:,k],area*w[k])
        return out

    def refine(self, selected):
        selected=np.asarray(selected,dtype=int)
        if len(np.unique(selected))!=len(selected):raise ValueError('duplicate selected cell')
        chosen=self.cells[selected]
        if (chosen[:,2]<4).any():raise ValueError('minimum cell size reached')
        children=[]
        for x,y,step in chosen:
            half=step//2
            children.extend([(x+i*half,y+j*half,half) for i in (0,1) for j in (0,1)])
        self.cells=np.concatenate([np.delete(self.cells,selected,axis=0),np.asarray(children,dtype=int)])
