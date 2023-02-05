from scipy.integrate import odeint
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np 


"""
Defining constants for the system of 2 coupled second order ODEs which are basically the Euler-Lagrange equations associated with the generalized coordinates theta_1, theta_2 
"""
m1, m2 = 4, 2
l1, l2 = 5, 5
g = 9.8


""" 
Defining a system of 4 coupled first order ODEs which is equivalent to the system of 2 coupled second order ODEs that describes the motion of a double pendulum
"""
def dS_dt(S, t):
    t1, t2, w1, w2 = S
    dt1_dt = w1
    dt2_dt = w2
    dw1_dt = (-(l1*(m2**2)*(l2**3)*(w2**2)*np.sin(t1-t2))-(m2*g*l1*np.sin(t1)*(m1+m2)*(l2**2))-(0.5*np.sin(2*(t1-t2))*(m2*l1*l2*w1)**2)+(g*l1*np.sin(t2)*np.cos(t1-t2)*(m2*l2)**2)) / ((m2*(m1+m2)*(l1**2)*(l2**2))-((m2**2)*(l1**2)*(l2**2)*(np.cos(t1-t2))**2))
    dw2_dt = ((m2*(m1+m2)*l2*(l1**3)*(w1**2)*np.sin(t1-t2))-(m2*(m1+m2)*g*l2*np.sin(t2)*(l1**2))+(0.5*np.sin(2*(t1-t2))*(m2*l1*l2*w2)**2)+(m2*(m1+m2)*g*l2*np.sin(t1)*np.cos(t1-t2)*l1**2)) / ((m2*(m1+m2)*(l1**2)*(l2**2))-((m2**2)*(l1**2)*(l2**2)*(np.cos(t1-t2))**2))
    return np.array([dt1_dt, dt2_dt, dw1_dt, dw2_dt])


""" 
Using Runge-Kutta Fourth Order method, the above system of 4 coupled first order ODEs have been solved
"""
t = np.linspace(0,50,501)
S0 = np.array([np.pi,np.pi/2, 0, 0])
sol = odeint(dS_dt, S0, t)
theta_1, theta_2, omega_1, omega_2= sol.T[0], sol.T[1], sol.T[2], sol.T[3]
x1 = l1*np.sin(theta_1)
y1 = -l1*np.cos(theta_1)
x2 = l1*np.sin(theta_1) + l2*np.sin(theta_2)
y2 = -l1*np.cos(theta_1) - l2*np.cos(theta_2)


""" 
Initializing objects for animation
"""
fig = plt.figure()
ax=plt.axes(xlim=(-(2*l1)-2,(2*l1)+2),ylim=(-(2*l1)-2,12))
bar1 = plt.Line2D((0,x1[0]),(0,y1[0]),color="blue",lw=1)
bob1 = plt.Circle((x1[0],y1[0]),0.25, fc="red", ec="red")
bar2 = plt.Line2D((x1[0],x2[0]),(y1[0],y2[0]),color="blue",lw=1)
bob2 = plt.Circle((x2[0],y2[0]),0.25, fc="red", ec="red")


""" 
Defining all the frames for the animation
"""
def update(i):
    bar1 = plt.Line2D((0,x1[i]),(0,y1[i]),color="blue",lw=1)
    ax.add_line(bar1)
    bob1.center=(x1[i],y1[i])
    ax.add_patch(bob1)
    bar2 = plt.Line2D((x1[i],x2[i]),(y1[i],y2[i]),color="blue",lw=1)
    ax.add_line(bar2)
    bob2.center=(x2[i],y2[i])
    ax.add_patch(bob2)
    return bar1, bob1, bar2, bob2,


""" 
Calling the Animation Function
"""
anime = ani.FuncAnimation(fig, update, frames=len(t), interval=50, blit=True, repeat=True)
ax.set_title(r"When $m_1=4,m_2=2$, $l_1=l_2=5$, $\theta_{1}(t=0)=\pi$, $\theta_{2}(t=0)=\frac{\pi}{2}$", color="fuchsia")
fig.suptitle("Double Pendulum", color="fuchsia")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
fig.tight_layout()
plt.annotate("Courtesy of Rishikesh Jha",(l1+3.5,(-2*l1)-1.5),color="fuchsia")
plt.axis("scaled")
plt.show()