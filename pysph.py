import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.scheme import SchemeChooser
from pysph.sph.wc.basic import WCSPHScheme
from pysph.tools.geometry import remove_overlap_particles

class Dambreak2D(Application):
    def create_particles(self):
        dx = 0.02
        hdx = 1.2
        rho = 1000.0
        c0 = 10.0
        alpha = 0.1

        # Create particles for the water column
        x, y = np.mgrid[-0.1:0.1:dx, -0.1:0.3:dx]
        x = x.ravel()
        y = y.ravel()
        m = np.ones_like(x) * dx**2 * rho
        h = np.ones_like(x) * hdx * dx
        rho = np.ones_like(x) * rho
        p = np.zeros_like(x)
        cs = np.ones_like(x) * c0
        pa = get_particle_array(name='fluid', x=x, y=y, m=m, h=h, rho=rho, p=p, cs=cs)
        
        # Create boundary particles
        xb, yb = np.mgrid[-0.2:0.5:dx, -0.2:0.0:dx]
        xb = xb.ravel()
        yb = yb.ravel()
        mb = np.ones_like(xb) * dx**2 * rho
        hb = np.ones_like(xb) * hdx * dx
        rhob = np.ones_like(xb) * rho
        pb = np.zeros_like(xb)
        csb = np.ones_like(xb) * c0
        boundary = get_particle_array(name='boundary', x=xb, y=yb, m=mb, h=hb, rho=rhob, p=pb, cs=csb)

        return [pa, boundary]

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=2, rho0=1000.0, c0=10.0, h0=0.02*1.2, hdx=1.2, alpha=0.1
        )
        return SchemeChooser(default='wcsph', wcsph=s)

    def configure_scheme(self):
        scheme = self.scheme
        scheme.configure_solver(dt=1e-4, tf=2.0, adaptive_timestep=True)

if __name__ == '__main__':
    app = Dambreak2D()
    app.run()
