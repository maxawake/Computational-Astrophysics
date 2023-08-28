import matplotlib.pyplot as plt
import matplotlib.animation as ani
import mpl_toolkits.axes_grid1
import scipy.fftpack as fp
import numpy as np

AE=149.6e9
v_earth=29.29e3/AE
sun_mass=1.9891e30
earth_mass=5.974e24
G=6.67428e-11/(AE**3)*sun_mass

def simple_rho(x,m,box_length,boxes):
        h=box_length/(boxes-1)
        x_new=np.array(np.round(x/h),dtype=int)
        grid=np.zeros((boxes,boxes))
        for i in range(np.size(m)):
                grid[x_new[i,0],x_new[i,1]]+=m[i]
        grid/=h**2
        return grid

def clouds_in_cells(x,m,box_length,boxes):
        h=box_length/(boxes)
        x_new=np.array(np.floor(x/h),dtype=int)
        grid=np.zeros((boxes,boxes))
        x_new=x_new%boxes

        for i in range(np.size(x_new,0)):
                grid[x_new[i,0],x_new[i,1]]+=m[i]
                grid[ ( x_new[ i ,0 ] + 1 ) % boxes , x_new[ i , 1 ] ] += m[i]
                grid[ x_new[ i , 0 ] , ( x_new[ i , 1 ] + 1 ) % boxes ] += m[i]
                grid[ ( x_new[ i , 0 ] + 1 ) % boxes , ( x_new[ i , 1 ] + 1 ) % boxes ] += m[i]
                
        grid/=h**2
        grid/=4.0
        return grid

def f_v(x,m,phi,h):#acceleration
        f=x.copy()
        boxes=np.size(phi,0)
        box_length=h*boxes

        index_x=np.array(np.floor(x[:,0]/h),dtype=int)
        index_y=np.array(np.floor(x[:,1]/h),dtype=int)
        index_x=(index_x)%boxes
        index_y=(index_y)%boxes
        index_x1=(index_x+1)%boxes
        index_y1=(index_y+1)%boxes

        f[:,0]=-((phi[index_x,index_y]-phi[index_x1,index_y])+(phi[index_x,index_y1]-phi[index_x1,index_y1]))
        f[:,1]=-((phi[index_x,index_y]-phi[index_x,index_y1])+(phi[index_x1,index_y]-phi[index_x1,index_y1]))
        f/=-2.0*h

        return f
        
def RK(x,v,m,phi,h,dt):
        x1=v
        v1=f_v(x,m, phi,h)
        
        x2=v+0.5*dt*v1
        v2=f_v(x+0.5*dt*x1,m, phi,h)
        
        x3=v+0.5*dt*v2
        v3=f_v(x+0.5*dt*x2,m,phi,h)
        
        x4=v+dt*v3
        v4=f_v(x+0.5*dt*x3,m,phi,h)
        
        xnew=x+dt/6.*(x1+2.*x2+2.*x3+x4)
        vnew=v+dt/6.*(v1+2.*v2+2.*v3+v4)
        return xnew,vnew
        

def leap_frog(x,v,m,phi,h,dt):
        v=v+(f_v(x,m, phi,h))*dt
        x=x+(v*dt)
        return x,v

def euler_step(x,v,m,phi,h,dt):
        xnew=x+dt*v
        vnew=(v+(f_v(x,m,phi,h))*dt)
        return xnew,vnew

class PM(object):
    def __init__(self,x0,v0,m0,box_length,boxes,dt,func):
        self.box_length = box_length
        self.boxes = boxes
        self.h=box_length/(boxes*1.0)
        self.x0=x0.copy()
        self.v0=v0.copy()
        self.m=m0.copy()
        self.dt = dt
        self.func = func

        self.green_k=np.zeros((boxes,boxes))
        half_box=boxes//2
        for i in range(boxes):
            for j in range(boxes):
                k_square=((i-half_box)**2+(j-half_box)**2)
                if k_square!=0:
                    self.green_k[i,j]=1.0/k_square
        self.green_k[half_box,half_box]=0
        self.green_k*=-4.0*np.pi*G
        self.green_k=fp.ifftshift(self.green_k)

        self.reset_data()

    def potential(self):
        rho=clouds_in_cells(self.x, self.m, self.box_length, self.boxes)
        rho_k=fp.fftn(rho)
        rho_k=fp.fftshift(rho_k)
        return np.real(fp.ifftn(fp.ifftshift(fp.fftshift(self.green_k)*rho_k)))

    def reset_data(self):
        self.x=self.x0.copy()
        self.v=self.v0.copy()

        ## leap frog init
        if self.func is leap_frog:
            phi = self.potential()
            self.x=self.x+0.5*(self.v.T*self.dt).T+0.125*(f_v(self.x,self.m, phi, self.h).T*(self.dt**2)).T
            self.x=self.x%box_length
             
    def init_anim(self):
        self.reset_data()
        ax1=self.fig.add_subplot(121)
        ax1.set_title('Potential')
        phi = self.potential()
        self.pic1=ax1.imshow(phi.T,origin='lower',interpolation='nearest',animated=False)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(self.pic1, cax=self.cax)
        cbar.solids.set_edgecolor('face')
        ax2=self.fig.add_subplot(122)
        ax2.set_autoscale_on(False)
        ax2.axis([0.0,self.box_length,0.0,self.box_length])
        ax2.set_facecolor('black')
        ax2.set_title('Spatial Distribution')
        self.pic2=ax2.scatter(self.x[:,0],self.x[:,1],marker='o',s=self.m*100,edgecolors='white',facecolors='white',animated=False)
        ax1.set_xticklabels('1')
        ax1.set_yticklabels('1')
        ax2.set_xlabel('x in AU')
        ax2.set_ylabel('y in AU')

        return [self.pic1, self.pic2, self.cax]

    def anim(self,days=10.,**kwargs):
        self.fig=plt.figure(figsize=(10,4))
        return ani.FuncAnimation(self.fig,self.update, init_func=self.init_anim, frames=int(np.ceil(days * 86400. / self.dt)), **kwargs)
            
    def update(self,i):
        dt_real=self.dt/60./60./24.
        self.fig.suptitle('t = %.1f d' % (i*dt_real))
        phi = self.potential()

        self.x,self.v=self.func(self.x,self.v,self.m,phi,self.h,self.dt)
        self.x=self.x%self.box_length

        self.pic2.set_offsets(self.x)
        if i%1==0: 
            self.pic1.set_array(phi.T)
            self.pic1.autoscale()
            return [self.pic1, self.pic2, self.cax]
        else:
            return [self.pic2]

    @classmethod
    def exercise5_3(cls, n_part=1000, m_part=1e-3, zero_padding=0.0, radial_distribution=False, box_length=1.0, boxes=64, dt=0.1, scheme='runge-kutta'):
        """convenience function for exercise 2 of problem set 5

        n_part:              number of particles
        m_part:              mass of a single particle
        zero_padding:        zero-padding of the particle distribution
        radial_distribution: only place particles inside a certain radius
        box_length:          size of the simulation box
        boxes:               mesh size
        dt:                  time step
        scheme:              time-integration scheme (can be runge-kutta, euler, leap-frog)
        """
        x=np.random.rand(n_part,2)*box_length
        if radial_distribution:
            x-=box_length/2
            for i in range(n_part):
                while np.linalg.norm(x[i])>box_length*0.5:
                    x[i]=np.random.rand(2)*box_length-box_length*0.5
            x+=box_length*0.5
        x+=0.5*zero_padding
        box_length+=zero_padding
        v=np.zeros((n_part,2))
        schemes = {'runge-kutta': RK, 'euler': euler_step, 'leap-frog': leap_frog}

        return cls(x, v, m_part * np.ones(n_part), box_length, boxes, dt * 86400., schemes[scheme])
