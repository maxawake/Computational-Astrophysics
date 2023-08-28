import numpy
from matplotlib.pyplot import *
import math
import os
from os import path



pi =  math.pi
B0=(1./4./pi)**0.5

i_rho  = 0
i_rhou = 1
i_rhov = 2
i_rhoE = 3
i_Bx   = 4
i_By   = 5

i_u    = 1
i_v    = 2
i_p    = 3

rc_const       = 0
rc_pwl         = 1


divB_none     = 0
divB_CCCT     = 1

t_rk1 = 0
t_rk2 = 1
t_rk3 = 2

bc_periodic   = 1

rusanov = 0


#----------------------------------------------------------------------# TIME INTEGRATOR




def fast_cfl_dt(g,U,p):
	

	apply_bc(g,U)
	p  = prim_from_state(g,U)
	
	

	Cfs = cfl_fast(g,U[:,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1],p[:,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1])

	u = (U[i_rhou,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]**2+U[i_rhov,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]**2)**0.5/U[i_rho,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]

	S = u + Cfs
	
	return g.cfl*min(g.dx,g.dy)/numpy.max(S)






def do_timestep(g,Un):



	if g.time_integrator == t_rk1:

		rk1(g,Un)

	elif g.time_integrator == t_rk2:

		rk2(g,Un)

	elif g.time_integrator == t_rk3:

		rk3(g,Un)
	
	

	#####################################

	if g.divB_control == divB_CCCT:

		CCCT(g,Un)

	#####################################


def rk1(g,Un):



	#1
	Res(g,Un)
	k1 = g.R[:,:,:]*g.dt
	update(g,Un,k1)



def rk2(g,Un):



	#1
	Res(g,Un)
	k1 = g.R[:,:,:]*g.dt
	update(g,Un,k1/2)


	#2
	Res(g,Un)
	k2 = g.R[:,:,:]*g.dt
	update(g,Un,k2)







def rk3(g,Un):



	#1
	Res(g,Un)
	k1 = g.R[:,:,:]*g.dt
	update(g,Un,k1/2)


	#2
	Res(g,g.U)
	k2 = g.R[:,:,:]*g.dt

	update(g,Un,2*k2-k1)

	#3
	Res(g,g.U)
	k3 = g.R[:,:,:]*g.dt
	update(g,Un,k3)

	
	#final update: 
	update(g,Un,1./6.*(k1+4*k2+k3))



def update(g,Un,k):

	
	g.U[:,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] = Un[:,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] - k[:,:,:]
	g.p  = prim_from_state(g,g.U)


#----------------------------------------------------------------------# RESIDUAL



def sweep(g,U):



		
		if g.rc_type == rc_const: #
			const(g,U)

		elif g.rc_type == rc_pwl: #
			pwl(g,U)

	

def Res(g,U):


	sweep(g,U)
	rusanov(g)	
	g.R[:,:,:] = g.R[:,:,:]/g.V






#----------------------------------------------------------------------# RECONSTRUCTION





def pwl(g,U):

		
	
	U = apply_bc(g,U)
	p = prim_from_state(g,U)
	

	#x-dir
					
	g.sigb_x = (p[:,g.ilowy:g.iuppy+1,g.ig1bx:g.ig2ax+1] -  p[:,g.ilowy:g.iuppy+1,g.ig1bx-1:g.ig2ax])/g.dx	
	g.sigf_x = (p[:,g.ilowy:g.iuppy+1,g.ilowx:g.ig2bx+1] -  p[:,g.ilowy:g.iuppy+1,g.ilowx-1:g.ig2bx])/g.dx



	i1 = numpy.where( ( numpy.abs(g.sigb_x) > numpy.abs(g.sigf_x) ) & (g.sigb_x*g.sigf_x>0) )
	i2 = numpy.where( ( numpy.abs(g.sigb_x) < numpy.abs(g.sigf_x) ) & (g.sigb_x*g.sigf_x>0) )
			

	g.sig_x[:,:,:] = 0.
	g.sig_x[i1] = g.sigf_x[i1]
	g.sig_x[i2] = g.sigb_x[i2] 

	g.pL_x = p[:,g.ilowy:g.iuppy+1,g.ig1bx:g.iuppx+1]  + g.sig_x[:,:,:-1]*g.dx/2.
	g.pR_x = p[:,g.ilowy:g.iuppy+1,g.ilowx:g.ig2ax+1]  - g.sig_x[:,:, 1:]*g.dx/2.

	
	g.pL_x[i_Bx] = (U[i_Bx,g.ilowy:g.iuppy+1,g.ig1bx:g.iuppx+1]+U[i_Bx,g.ilowy:g.iuppy+1,g.ilowx:g.ig2ax+1])/2.
	g.pR_x[i_Bx] = (U[i_Bx,g.ilowy:g.iuppy+1,g.ig1bx:g.iuppx+1]+U[i_Bx,g.ilowy:g.iuppy+1,g.ilowx:g.ig2ax+1])/2.


	#y-dir

	g.sigb_y = (p[:,g.ig1by:g.ig2ay+1,g.ilowx:g.iuppx+1] -  p[:,g.ig1by-1:g.ig2ay,g.ilowx:g.iuppx+1])/g.dy	
	g.sigf_y = (p[:,g.ilowy:g.ig2by+1,g.ilowx:g.iuppx+1] -  p[:,g.ilowy-1:g.ig2by,g.ilowx:g.iuppx+1])/g.dy

			
	i1 = numpy.where( ( numpy.abs(g.sigb_y) > numpy.abs(g.sigf_y) ) & (g.sigb_y*g.sigf_y>0) )
	i2 = numpy.where( ( numpy.abs(g.sigb_y) < numpy.abs(g.sigf_y) ) & (g.sigb_y*g.sigf_y>0) )
			

	g.sig_y[:,:,:] = 0.
	g.sig_y[i1] = g.sigf_y[i1]
	g.sig_y[i2] = g.sigb_y[i2] 


	g.pL_y = p[:,g.ig1by:g.iuppy+1,g.ilowx:g.iuppx+1]  + g.sig_y[:,:-1,:]*g.dy/2.
	g.pR_y = p[:,g.ilowy:g.ig2ay+1,g.ilowx:g.iuppx+1]  - g.sig_y[:, 1:,:]*g.dy/2.



	g.pL_y[i_By] = (U[i_By,g.ig1by:g.iuppy+1,g.ilowx:g.iuppx+1]+U[i_By,g.ilowy:g.ig2ay+1,g.ilowx:g.iuppx+1])/2.
	g.pR_y[i_By] = (U[i_By,g.ig1by:g.iuppy+1,g.ilowx:g.iuppx+1]+U[i_By,g.ilowy:g.ig2ay+1,g.ilowx:g.iuppx+1])/2.

	
	#update

	g.UL_x = state_from_prim(g,g.pL_x)
	g.UR_x = state_from_prim(g,g.pR_x)			
	g.UL_y = state_from_prim(g,g.pL_y)
	g.UR_y = state_from_prim(g,g.pR_y)








def const(g,U):


		U = apply_bc(g,U)
		p = prim_from_state(g,U)


		#x-dir

		g.pL_x = p[:,g.ilowy:g.iuppy+1,g.ig1ax:g.iuppx+1]
		g.pR_x = p[:,g.ilowy:g.iuppy+1,g.ilowx:g.ig2ax+1]


		#y-dir

		g.pL_y = p[:,g.ig1ay:g.iuppy+1,g.ilowx:g.iuppx+1]
		g.pR_y = p[:,g.ilowy:g.ig2ay+1,g.ilowx:g.iuppx+1]


		#update

		g.UL_x = state_from_prim(g,g.pL_x)
		g.UR_x = state_from_prim(g,g.pR_x)
		g.UL_y = state_from_prim(g,g.pL_y)
		g.UR_y = state_from_prim(g,g.pR_y)



	




#----------------------------------------------------------------------# OUTPUT



def write_output(g,filename='none'):



		Nxs     = g.Nx
		Nys     = g.Ny
		ts     = g.time

		xs     = g.x[g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		ys     = g.y[g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		
		rhos   = g.U[i_rho,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		rhous  = g.U[i_rhou,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		rhovs  = g.U[i_rhov,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		rhoEs  = g.U[i_rhoE,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		Bxs    = g.U[i_Bx,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		Bys    = g.U[i_By,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]

		ps     = g.p[i_p,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]
		machs  = g.mach()[g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1]

		
		divBs    = g.divB_central()
		rerr_divBs =  numpy.log10(numpy.mean(g.rel_divB()))

		if filename == 'none':
		

			numpy.savez('%s/%s'%(g.fdr,g.ic), Nx = Nxs, Ny = Nys,t = ts, x = xs, y = ys, rho = rhos, rhou = rhous, rhov = rhovs, rhoE = rhoEs, Bx = Bxs, By = Bys, p = ps, mach = machs, divB = divBs, rerr_divB = rerr_divBs )


		else:

			numpy.savez('%s/%s'%(g.fdr,filename), Nx = Nxs, Ny = Nys,t = ts, x = xs, y = ys, rho = rhos, rhou = rhous, rhov = rhovs, rhoE = rhoEs, Bx = Bxs, By = Bys, p = ps, mach = machs, valf = valfs, divB = divBs, rerr_divB = rerr_divBs )


		g.check_noutput = g.check_noutput + 1

		if (g.check_noutput > 500 ):

			g.foutput = 1000000
            
            
            
            


class output:

		
	def __init__(self, filename):

			
		npz = numpy.load(filename)


		self.Nx = npz['Nx']
		self.Ny = npz['Ny']
		self.t = npz['t']
		self.x = npz['x']
		self.y = npz['y']
		self.rho  = npz['rho']
		self.rhou = npz['rhou']
		self.rhov = npz['rhov']
		self.rhoE = npz['rhoE']
		self.Bx   = npz['Bx']
		self.By   = npz['By']
		self.p    = npz['p']
		self.mach = npz['mach']
		self.divB = npz['divB']
		self.rerr_divB = npz['rerr_divB']






#----------------------------------------------------------------------# GRID




class grid:

		
	def __init__(self,Nx,Ny,x1,x2,y1,y2,bcx_boundary,bcy_boundary,cfl,tmax,gamma,rc_type,time_integrator,divB_control,fdr):
	


		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2


	

		
		self.dx = (x2-x1)/Nx
		self.dy = (y2-y1)/Ny
		self.Nx = Nx
		self.Ny = Ny
		self.rc_type = rc_type
		self.divB_control = divB_control
		self.time_integrator = time_integrator
		self.gamma = gamma
		self.cfl = cfl
		self.foutput = 100000
		self.fdr = fdr
		self.bcx_boundary = bcx_boundary
		self.bcy_boundary = bcy_boundary
        
		self.tmax = tmax

		self.Ax = self.dy
		self.Ay = self.dx
		self.V = self.dx*self.dy
		self.dt = 0.
		self.time = 0.
		self.ic = 0
		self.check_noutput = 0
		

	

		if ( rc_type == rc_const ):
			
			
			self.ngc = 1
			self.ig1ax = 0
			self.ig2ax = (self.Nx+self.ngc) 
			self.ig1ay = 0
			self.ig2ay = (self.Ny+self.ngc) 


		elif ( rc_type == rc_pwl ): 
			
			self.ngc = 2


			self.ig1ax = 0
			self.ig1bx = 1
			self.ig2ax = (self.Nx+self.ngc) 
			self.ig2bx = self.ig2ax+1



			self.ig1ay = 0
			self.ig1by = 1
			self.ig2ay = (self.Ny+self.ngc) 
			self.ig2by = self.ig2ay+1


			self.sigf_x,self.sigb_x,self.sigc_x,self.sig_x = self.sig_linear_rec_x()
			self.sigf_y,self.sigb_y,self.sigc_y,self.sig_y = self.sig_linear_rec_y()
			
		


		self.ilowx = self.ngc
		self.iuppx = (self.Nx+self.ngc) - 1

		self.ilowy = self.ngc
		self.iuppy = (self.Ny+self.ngc) - 1


		self.x,self.y = self.cartesian_2D()
		self.U = self.cons()
		self.p = self.prim()


		self.R = self.residual()
		self.fx_L,self.fy_L = self.flux_left()
		self.fx_R,self.fy_R = self.flux_right()

		self.UL_x,self.UL_y = self.U_left()
		self.UR_x,self.UR_y = self.U_right()

	
		self.pL_x,self.pL_y = self.p_left()
		self.pR_x,self.pR_y = self.p_right()

	

	
	def cartesian_2D(self):
		
		
		x = numpy.ones((self.Ny+2*self.ngc,self.Nx+2*self.ngc))
		y = numpy.ones((self.Ny+2*self.ngc,self.Nx+2*self.ngc))
			
		x_c = numpy.linspace(self.x1+self.dx/2.,self.x2-self.dx/2.,self.Nx)
		y_c = numpy.linspace(self.y1+self.dy/2.,self.y2-self.dy/2.,self.Ny)

		x_c,y_c = numpy.meshgrid(x_c,y_c)

		x[self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1] = x_c
		y[self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1] = y_c
		

		return x,y

	def cons(self):
		
		U = numpy.ones((6,self.Ny+2*self.ngc,self.Nx+2*self.ngc))
		return U

	def prim(self):

		U = numpy.ones((6,self.Ny+2*self.ngc,self.Nx+2*self.ngc))
		return U

	def residual(self):

		R  = numpy.ones((6,self.Ny,self.Nx))
		return R

	def flux_left(self):

		fx_L = numpy.ones((6,self.Ny,self.Nx+1))
		fy_L = numpy.ones((6,self.Ny+1,self.Nx))
	

		return fx_L,fy_L


	def flux_right(self):

		fx_R = numpy.ones((6,self.Ny,self.Nx+1))
		fy_R = numpy.ones((6,self.Ny+1,self.Nx))
		

		return fx_R,fy_R


	def U_left(self):
	
		U_Lx = numpy.ones((6,self.Ny,self.Nx+1))
		U_Ly = numpy.ones((6,self.Ny+1,self.Nx))

		return U_Lx,U_Ly

	def U_right(self):
	
		U_Rx = numpy.ones((6,self.Ny,self.Nx+1))
		U_Ry = numpy.ones((6,self.Ny+1,self.Nx))

		return U_Rx,U_Ry


	def p_right(self):

		p_Lx = numpy.ones((6,self.Ny,self.Nx+1))
		p_Ly = numpy.ones((6,self.Ny+1,self.Nx))

		return p_Lx,p_Ly

	
	def p_left(self):

		p_Rx = numpy.ones((6,self.Ny,self.Nx+1))
		p_Ry = numpy.ones((6,self.Ny+1,self.Nx))

		return p_Rx,p_Ry


	def sig_linear_rec_x(self):

		sigb = numpy.ones((6,self.Ny,self.Nx+2))
		sigf = numpy.ones((6,self.Ny,self.Nx+2))
		sigc = numpy.ones((6,self.Ny+2,self.Nx))
		sig  = numpy.ones((6,self.Ny,self.Nx+2))

		return sigb,sigf,sigc,sig


	def sig_linear_rec_y(self):

		sigb = numpy.ones((6,self.Ny+2,self.Nx))
		sigf = numpy.ones((6,self.Ny+2,self.Nx))
		sigc = numpy.ones((6,self.Ny+2,self.Nx))
		sig  = numpy.ones((6,self.Ny+2,self.Nx))


		return sigb,sigf,sigc,sig
		


	def mach(self):


		self.U = apply_bc(self,self.U)
		self.p = apply_bc(self,self.p)

		cs    = (self.gamma*self.p[i_p,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]/self.U[i_rho,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1])**0.5
		v_cs  = (self.U[i_rhou,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]**2+self.U[i_rhov,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]**2)**0.5/self.U[i_rho,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]/cs

		return v_cs




	
	def divB_central(self):


		self.U = apply_bc(self,self.U)

		divBx = (self.U[i_Bx,self.ilowy:self.iuppy+1,self.ilowx+1:self.ig2ax+1] - self.U[i_Bx,self.ilowy:self.iuppy+1,self.ilowx-1:self.ig2ax-1])/self.dx/2.
		divBy = (self.U[i_By,self.ilowy+1:self.ig2ay+1,self.ilowx:self.iuppx+1] - self.U[i_By,self.ilowy-1:self.ig2ay-1,self.ilowx:self.iuppx+1])/self.dy/2.
		divB = (divBx+divBy)


		return divB




	def rel_divB(self):

		self.U = apply_bc(self,self.U)
		
		
		Bbar = (self.U[i_Bx,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]**2+self.U[i_By,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1]**2)**0.5
		err =  self.divB_central()*(self.dx**2+self.dy**2)**0.5/Bbar
		
		return numpy.abs(err)



	def Etot(self):


		self.U = apply_bc(self,self.U)

		return numpy.sum( self.U[i_rhoE,self.ilowy:self.iuppy+1,self.ilowx:self.iuppx+1] )

	







#----------------------------------------------------------------------# FUNCTIONS




def prim_from_state(g,U):

		p = U.copy()
		
		
		p[i_u,:,:]   = U[i_rhou,:,:]/U[i_rho,:,:]
		p[i_v,:,:]   = U[i_rhov,:,:]/U[i_rho,:,:]
		p[i_p,:,:] = (g.gamma-1.)*( U[i_rhoE,:,:] - 0.5*(U[i_rhou,:,:]**2+U[i_rhov,:,:]**2)/U[i_rho,:,:] - 0.5*(U[i_Bx,:,:]**2+U[i_By,:,:]**2) )
		
		return p


def state_from_prim(g,p):

		U = p.copy()
		
		U[i_rhou,:,:]   = p[i_u,:,:]*p[i_rho,:,:]
		U[i_rhov,:,:]   = p[i_v,:,:]*p[i_rho,:,:]
		U[i_rhoE,:,:]   = p[i_p,:,:]/(g.gamma-1.) + 0.5*(p[i_u,:,:]**2+p[i_v,:,:]**2)*p[i_rho,:,:] + 0.5*(p[i_Bx,:,:]**2+p[i_By,:,:]**2) 
		
		return U



def cfl_fast(g,U,p):
	
	
	cs        =  (g.gamma*p[i_p,:,:]/U[i_rho,:,:])**0.5
	a         =  ((U[i_Bx,:,:]**2+U[i_By,:,:]**2)/U[i_rho,:,:])**0.5



	S = (cs**2+a**2)**0.5 

	return S



def cfl_fastx(g,U,p):
	
	

	cs        =  (g.gamma*p[i_p,:,:]/U[i_rho,:,:])**0.5
	a         =  ((U[i_Bx,:,:]**2+U[i_By,:,:]**2)/U[i_rho,:,:])**0.5
	ax         =  ((U[i_Bx,:,:]**2)/U[i_rho,:,:])**0.5
	Sx         = ( 0.5*( (cs**2+a**2) + ( (cs**2+a**2)**2 - 4*ax**2*cs**2 )**0.5  ) )**0.5 
	
	return Sx




def cfl_fasty(g,U,p):
	
	cs        =  (g.gamma*p[i_p,:,:]/U[i_rho,:,:])**0.5
	a         =  ((U[i_Bx,:,:]**2+U[i_By,:,:]**2)/U[i_rho,:,:])**0.5
	ay         =  ((U[i_By,:,:]**2)/U[i_rho,:,:])**0.5
	Sy         = ( 0.5*( (cs**2+a**2) + ( (cs**2+a**2)**2 - 4*ay**2*cs**2 )**0.5  ) )**0.5 
	
	return Sy





def check(g):


		ratio = g.time/g.tmax*100
		print('%.4f %%'%(ratio))








#----------------------------------------------------------------------# FLUX






def MHD_xflux(F,U,p):


		F[i_rho,:,:]   = U[i_rhou,:,:]
		F[i_rhou,:,:]  = U[i_rhou,:,:]**2/U[i_rho,:,:] + p[i_p,:,:] + 0.5*(U[i_Bx,:,:]**2 + U[i_By,:,:]**2) - U[i_Bx,:,:]**2
		F[i_rhov,:,:]  = U[i_rhou,:,:]*U[i_rhov,:,:]/U[i_rho,:,:] - U[i_Bx,:,:]*U[i_By,:,:]
		F[i_rhoE,:,:]  = (U[i_rhoE,:,:]+p[i_p,:,:]+0.5*(U[i_Bx,:,:]**2 + U[i_By,:,:]**2))*U[i_rhou,:,:]/U[i_rho,:,:] - (U[i_Bx,:,:]*U[i_rhou,:,:]+U[i_By]*U[i_rhov])/U[i_rho]*U[i_Bx,:,:]
		F[i_Bx,:,:]    = 0.
		F[i_By,:,:]    = U[i_rhou,:,:]*U[i_By,:,:]/U[i_rho,:,:] - U[i_rhov,:,:]*U[i_Bx,:,:]/U[i_rho,:,:]




def MHD_yflux(F,U,p):



		F[i_rho,:,:]   = U[i_rhov,:,:]
		F[i_rhou,:,:]  = U[i_rhou,:,:]*U[i_rhov,:,:]/U[i_rho,:,:] - U[i_Bx,:,:]*U[i_By,:,:] 
		F[i_rhov,:,:]  = U[i_rhov,:,:]**2/U[i_rho,:,:] + p[i_p,:,:] + 0.5*(U[i_Bx,:,:]**2 + U[i_By,:,:]**2) - U[i_By,:,:]**2
		F[i_rhoE,:,:]  = (U[i_rhoE,:,:]+p[i_p,:,:]+0.5*(U[i_Bx,:,:]**2 + U[i_By,:,:]**2))*U[i_rhov,:,:]/U[i_rho,:,:] - (U[i_Bx,:,:]*U[i_rhou,:,:]+U[i_By,:,:]*U[i_rhov,:,:])/U[i_rho,:,:]*U[i_By,:,:]
		F[i_Bx,:,:]    = U[i_rhov,:,:]*U[i_Bx,:,:]/U[i_rho,:,:] - U[i_rhou,:,:]*U[i_By,:,:]/U[i_rho,:,:]
		F[i_By,:,:]    = 0.





def rusanov(g):



  
	########################################## X-DIR

	MHD_xflux(g.fx_L,g.UL_x,g.pL_x)
	MHD_xflux(g.fx_R,g.UR_x,g.pR_x)

	CFL = cfl_fastx(g,g.UL_x,g.pL_x)
	CFR = cfl_fastx(g,g.UR_x,g.pR_x)

	lamx_L = abs(g.UL_x[i_rhou,:,:]/g.UL_x[i_rho,:,:]) + CFL
	lamx_R = abs(g.UR_x[i_rhou,:,:]/g.UR_x[i_rho,:,:]) + CFR
	a = 0.5*(lamx_L+lamx_R)
       
	fx = 0.5*g.Ax*(  g.fx_L[:,:,:]+g.fx_R[:,:,:] -a*(g.UR_x[:,:,:]-g.UL_x[:,:,:])  )
	g.R = fx[:,:,1:] - fx[:,:,:-1]
	


	########################################## Y-DIR

	MHD_yflux(g.fy_L,g.UL_y,g.pL_y)
	MHD_yflux(g.fy_R,g.UR_y,g.pR_y)

	CFL = cfl_fasty(g,g.UL_y,g.pL_y)
	CFR = cfl_fasty(g,g.UR_y,g.pR_y)


	lamy_L = abs(g.UL_y[i_rhov,:,:]/g.UL_y[i_rho,:,:]) + CFL
	lamy_R = abs(g.UR_y[i_rhov,:,:]/g.UR_y[i_rho,:,:]) + CFR
	a = 0.5*(lamy_L+lamy_R)

	fy = 0.5*g.Ay*(  g.fy_L[:,:,:]+g.fy_R[:,:,:] -a*(g.UR_y[:,:,:]-g.UL_y[:,:,:])  )
	g.R[:,:,:] = g.R[:,:,:] + ( fy[:,1:,:] - fy[:,:-1,:])



#----------------------------------------------------------------------# CCCT




def CCCT(g,Un):

	U_star = apply_bc(g,g.U)

	pn     = prim_from_state(g,Un)
	p_star = prim_from_state(g,U_star)
	
	
	emf_n    =   (pn[i_u,:,:]*Un[i_By,:,:]-pn[i_v,:,:]*Un[i_Bx,:,:])
	emf_star =   (p_star[i_u,:,:]*U_star[i_By,:,:]-p_star[i_v,:,:]*U_star[i_Bx,:,:])
	emf = -(emf_n[:,:]+emf_star[:,:])/2.

	
	g.U[i_Bx,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] = Un[i_Bx,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] - g.dt*(emf[g.ilowy+1:g.ig2ay+1,g.ilowx:g.iuppx+1]-emf[g.ilowy-1:g.ig2ay-1,g.ilowx:g.iuppx+1])/2./g.dy 
	g.U[i_By,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] = Un[i_By,g.ilowy:g.iuppy+1,g.ilowx:g.iuppx+1] + g.dt*(emf[g.ilowy:g.iuppy+1,g.ilowx+1:g.ig2ax+1]-emf[g.ilowy:g.iuppy+1,g.ilowx-1:g.ig2ax-1])/2./g.dx 




#----------------------------------------------------------------------# BOUNDARY


def apply_bc(g,q):

	
	#x

	if g.bcx_boundary==bc_periodic:

		
		q =  bcx_per(g, q)


	#y


	if g.bcy_boundary==bc_periodic:

		
		q =  bcy_per(g, q)

	return q

		

def bcx_per(g, q):


	

	if g.ngc == 1:

		

	
		q[:,g.ilowy:g.iuppy+1,g.ig1ax] = q[:,g.ilowy:g.iuppy+1,g.iuppx]
		q[:,g.ilowy:g.iuppy+1,g.ig2ax] = q[:,g.ilowy:g.iuppy+1,g.ilowx]


	elif g.ngc ==2:

		
	
		q[:,g.ilowy:g.iuppy+1,g.ig1ax] = q[:,g.ilowy:g.iuppy+1,g.iuppx-1]
		q[:,g.ilowy:g.iuppy+1,g.ig1bx] = q[:,g.ilowy:g.iuppy+1,g.iuppx]
		q[:,g.ilowy:g.iuppy+1,g.ig2ax] = q[:,g.ilowy:g.iuppy+1,g.ilowx]
		q[:,g.ilowy:g.iuppy+1,g.ig2bx] = q[:,g.ilowy:g.iuppy+1,g.ilowx+1]

	return q
		

def bcy_per(g, q):


	

	if g.ngc == 1:

		

		q[:,g.ig1ay,g.ilowx:g.iuppx+1] = q[:,g.iuppy,g.ilowx:g.iuppx+1]
		q[:,g.ig2ay,g.ilowx:g.iuppx+1] = q[:,g.ilowy,g.ilowx:g.iuppx+1]
		

	elif g.ngc ==2:

		
	
		q[:,g.ig1ay,g.ilowx:g.iuppx+1] = q[:,g.iuppy-1,g.ilowx:g.iuppx+1]
		q[:,g.ig1by,g.ilowx:g.iuppx+1] = q[:,g.iuppy,g.ilowx:g.iuppx+1]
		q[:,g.ig2ay,g.ilowx:g.iuppx+1] = q[:,g.ilowy,g.ilowx:g.iuppx+1]
		q[:,g.ig2by,g.ilowx:g.iuppx+1] = q[:,g.ilowy+1,g.ilowx:g.iuppx+1]
	
	return q
		

#----------------------------------------------------------------------# ORSZAG-TANG VORTEX SETUP	


def orszag_tang(inputs):


	Nx,Ny,t_max,cfl,rc_type,time_integrator,divB_control,fdr = inputs 

	

	bcx_boundary = bc_periodic
	bcy_boundary = bc_periodic
	gamma = 5./3.



		
	x1 = 0.
	x2 = 1.
	y1 = 0.
	y2 = 1.


	g = grid(Nx,Ny,x1,x2,y1,y2,bcx_boundary,bcy_boundary,cfl,t_max,gamma,rc_type,time_integrator,divB_control,fdr)





	g.p[i_p,:,:]    =  5./12./pi
	g.U[i_rho,:,:]  =  25./36./pi
	g.U[i_rhou,:,:] = -g.U[i_rho,:,:]*numpy.sin(2*pi*g.y[:,:]) 
	g.U[i_rhov,:,:] =  g.U[i_rho,:,:]*numpy.sin(2*pi*g.x[:,:]) 
	g.U[i_Bx,:,:]   = -B0*numpy.sin(2*pi*g.y[:,:])
	g.U[i_By,:,:]   =  B0*numpy.sin(4*pi*g.x[:,:])
	g.U[i_rhoE,:,:] = g.p[i_p,:,:]/(g.gamma-1.) + 0.5*(g.U[i_rhou,:,:]**2+g.U[i_rhov,:,:]**2)/g.U[i_rho,:,:] + 0.5*(g.U[i_Bx,:,:]**2+g.U[i_By,:,:]**2)




	dt_guess = 0.5*fast_cfl_dt(g,g.U,g.p)
	Ntot = int(t_max/dt_guess)

	g.foutput = int(Ntot/50)

	while(g.time<t_max):
		
			
		#######################

		check(g)

		#######################

		g.dt =  fast_cfl_dt(g,g.U,g.p)
		
		#######################

		if (g.dt+g.time)>t_max:

			g.dt = t_max-g.time

		if (g.ic % g.foutput == 0):

			write_output(g)
			
		#######################

		Un = (g.U).copy()
		do_timestep(g,Un)
		


		
		g.time = g.time+g.dt
		g.ic = g.ic + 1

  
	write_output(g)
	


	return g







