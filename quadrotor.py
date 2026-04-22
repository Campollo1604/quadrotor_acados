#Script que define la fisica del dron, su estado y como se mueve

from math import sqrt
import numpy as np
from utils import quaternion_to_euler, skew_symmetric, v_dot_q, unit_quat, quaternion_inverse


class Quadrotor3D:

    def __init__(self, noisy=False, drag=False, payload=False, motor_noise=False):
        #Añade posibles efectos físicos como la resistencia aerodinámica, carga de pago, perturbaciones externas como viento o incluso variaciones en el empuje de los motores
        """
        Initialization of the 3D quadrotor class
        :param noisy: Whether noise is used in the simulation
        :type noisy: bool
        :param drag: Whether to simulate drag or not.
        :type drag: bool
        :param payload: Whether to simulate a payload force in the simulation
        :type payload: bool
        :param motor_noise: Whether non-gaussian noise is considered in the motor inputs
        :type motor_noise: bool
        """

        configuration = 'x' #Configuracion en x, motores a 45º

        
        self.max_thrust = 20 #Máximo empuje por motor

        #Se define el estado completo del dron, un total de 13 componentes, todas comienzan en (0,0,0), menos el de orientación. Los cuaterniones de rotación siempre tienen módulo 1, en este caso es el de 
        #identidad y significa ninguna rotación
        self.pos = np.zeros((3,)) #Posición (x,y,z)
        self.vel = np.zeros((3,)) #Velocidad lineal (vx,vy,vz)
        self.angle = np.array([1., 0., 0., 0.])  #Cuaterniones: qw, qx, qy, qz
        self.a_rate = np.zeros((3,)) #Velocidades angulares (wx,wy,wz)

        #Se normaliza la potencia de los motores entre [0,1], de esta forma trabaja con porcentajes y no con Newtons
        self.max_input_value = 1  # Motores a potencia máxima
        self.min_input_value = 0  # Motors apagados

        #Parámetros internos del dron 
        self.J = np.array([.03, .03, .06])  #Tensor de inercia [kg m^2]. Valores corresponden con el valor en cada eje, Jx y Jy son iguales ya que el dron es simétrico en los ejes x e y. 
        #Jz es mayor ya que los 4 motores se encuentran a la mayor distancia del eje z
        self.mass = 1.0  #Masa [kg]

        #Distancia de los motores al centro de gravedad [m]
        self.length = 0.47 / 2  

        #Posición de los motores dependiendo de si es x o +
        if configuration == '+':
            self.x_f = np.array([self.length, 0, -self.length, 0]) #Como están alineados con los ejes x e y, están a la distancia que mide el brazo
            self.y_f = np.array([0, self.length, 0, -self.length])
        elif configuration == 'x':
            h = np.cos(np.pi / 4) * self.length
            self.x_f = np.array([h, -h, -h, h])  #Como los motores se encuentran a 45º, es neceario calcular la hipotenusa a partir del angulo (pi/4) y la distancia del brazo. 
            self.y_f = np.array([-h, -h, h, h]) #Como cos(45) y sen(45) tienen el mismo valor, x e y son iguales

        #Los motores al girar, a parte de empuje, generan un torque, que en vuelo estacionario se cancela
        self.c = 0.013  #Esta es una constante que al multiplicarla por el valor T de cada motor nos da el torque total. Para que haya giro sobre z, el torque debe ser distinto de 0   
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c]) #Motor 1 y 3 giran en un sentido (torque negativo) y motor 2 y 4 giran en el opuesto (torque positivo)

        #Vector gravedad [m s^-2]
        self.g = np.array([[0], [0], [9.81]])  

        #Actuación de los 4 motores en Newtons 
        self.u_noiseless = np.array([0.0, 0.0, 0.0, 0.0]) #Este simularía un empuje perfecto o teórico
        self.u = np.array([0.0, 0.0, 0.0, 0.0]) #Empuje "real", al activar motor_noise = True motor presentaría variaciones, perturbaciones etc

        #Coeficiente de drag [kg / m]
        self.rotor_drag_xy = 0.3 #Cuando el dron se mueve lateralmente y el aire golpea contra la hélices, se genera una fuerza contraria al movimiento
        self.rotor_drag_z = 0.0  #Las hélices no giran en este eje
        self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])[:, np.newaxis] #Transformación en columna para facilitar cálculos
        self.aero_drag = 0.08 #Roce del aire contra el dron. Determinará la velocidad máxima, cuando la fuerza aplicada sea igual a la resistencia del aire

        #Opciones para aumentar realismo a la simulación
        self.drag = drag
        self.noisy = noisy
        self.motor_noise = motor_noise

        self.payload_mass = 0.3  # kg
        self.payload_mass = self.payload_mass * payload #Si payload = True = 1, si no False = 0

    def set_state(self, *args, **kwargs): #Sirve para colocar al dron en una posición específica. Se puede hacer de dos formas, con un array de 13 números o con argumentos con nombres
        if len(args) != 0:
            assert len(args) == 1 and len(args[0]) == 13 #Entrada por lista. Se comprueba que has pasado los datos exactos
            self.pos[0], self.pos[1], self.pos[2], \
            self.angle[0], self.angle[1], self.angle[2], self.angle[3], \
            self.vel[0], self.vel[1], self.vel[2], \
            self.a_rate[0], self.a_rate[1], self.a_rate[2] \
                = args[0]

        else: #Entrada por nombre
            self.pos = kwargs["pos"]
            self.angle = kwargs["angle"]
            self.vel = kwargs["vel"]
            self.a_rate = kwargs["rate"]

    def get_state(self, quaternion=False, stacked=False): #Esta función devuelve la información en formatos diferentes. Cuaterniones o ángulos de Euler / Lista de arrays o lista plana

        if quaternion and not stacked:
            return [self.pos, self.angle, self.vel, self.a_rate] #Devuelve 4 arrays
        if quaternion and stacked:
            return [self.pos[0], self.pos[1], self.pos[2], self.angle[0], self.angle[1], self.angle[2], self.angle[3], #Devuelve el estado completo del dron (13)
                    self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]]

        angle = quaternion_to_euler(self.angle) #Transforma cuaterniones en ángulos de Euler
        if not quaternion and stacked:
            return [self.pos[0], self.pos[1], self.pos[2], angle[0], angle[1], angle[2], #En este caso 12 números al ser ángulos de Euler
                    self.vel[0], self.vel[1], self.vel[2], self.a_rate[0], self.a_rate[1], self.a_rate[2]]
        return [self.pos, angle, self.vel, self.a_rate]

    def get_control(self, noisy=False): #Saber si trabajamos en mundo teórico o real 
        if not noisy:
            return self.u_noiseless
        else:
            return self.u

    def update(self, u, dt):
        """
        Runge-Kutta 4th order dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
        :param dt: time differential
        """

        # Clip inputs
        for i, u_i in enumerate(u):
            self.u_noiseless[i] = max(min(u_i, self.max_input_value), self.min_input_value)

        # Apply noise to inputs (uniformly distributed noise with standard deviation proportional to input magnitude)
        if self.motor_noise:
            for i, u_i in enumerate(self.u_noiseless):
                std = 0.02 * sqrt(u_i)
                noise_u = np.random.normal(loc=0.1 * (u_i / 1.3) ** 2, scale=std)
                self.u[i] = max(min(u_i - noise_u, self.max_input_value), self.min_input_value) * self.max_thrust
        else:
            self.u = self.u_noiseless * self.max_thrust

        # Generate disturbance forces / torques
        if self.noisy:
            f_d = np.random.normal(size=(3, 1), scale=10 * dt)
            t_d = np.random.normal(size=(3, 1), scale=10 * dt)
        else:
            f_d = np.zeros((3, 1))
            t_d = np.zeros((3, 1))

        x = self.get_state(quaternion=True, stacked=False)

        # RK4 integration
        k1 = [self.f_pos(x), self.f_att(x), self.f_vel(x, self.u, f_d), self.f_rate(x, self.u, t_d)]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        k3 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        k4 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in
             range(4)]

        # Ensure unit quaternion
        x[1] = unit_quat(x[1])

        self.pos, self.angle, self.vel, self.a_rate = x

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = x[3]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = np.array([[0], [0], [np.sum(u)]]) / self.mass

        if self.drag:
            # Transform velocity to body frame
            v_b = v_dot_q(x[2], quaternion_inverse(x[1]))[:, np.newaxis]
            # Compute aerodynamic drag acceleration in world frame
            a_drag = -self.aero_drag * v_b ** 2 * np.sign(v_b) / self.mass
            # Add rotor drag
            a_drag -= self.rotor_drag * v_b / self.mass
            # Transform drag acceleration to world frame
            a_drag = v_dot_q(a_drag, x[1])
        else:
            a_drag = np.zeros((3, 1))

        angle_quaternion = x[1]

        a_payload = -self.payload_mass * self.g / self.mass

        return np.squeeze(-self.g + a_payload + a_drag + v_dot_q(a_thrust + f_d / self.mass, angle_quaternion))

    def f_rate(self, x, u, t_d):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param t_d: disturbance torque (3D)
        :return: angular rate differential increment (scalar): dr/dt
        """

        rate = x[3]
        return np.array([
            1 / self.J[0] * (u.dot(self.y_f) + t_d[0] + (self.J[1] - self.J[2]) * rate[1] * rate[2]),
            1 / self.J[1] * (-u.dot(self.x_f) + t_d[1] + (self.J[2] - self.J[0]) * rate[2] * rate[0]),
            1 / self.J[2] * (u.dot(self.z_l_tau) + t_d[2] + (self.J[0] - self.J[1]) * rate[0] * rate[1])
        ]).squeeze()
