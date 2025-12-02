import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**6 + 3*x**5 + 4*x**2 - 1

def df2(x): # Друга похідна
    return 60*x**4 + 60*x**3 + 8

x_nodes = np.array([1.0, 3.0, 5.0])
y_nodes = f(x_nodes)
h_step = 2.0 

c0 = df2(x_nodes[0]) # 128
c2 = df2(x_nodes[-1]) # 45008
rhs = 6 * ((y_nodes[2]-y_nodes[1])/h_step - (y_nodes[1]-y_nodes[0])/h_step)
lhs_known = h_step*c0 + h_step*c2
c1 = (rhs - lhs_known) / (2 * (h_step + h_step))
c_coeffs = [c0, c1, c2]

spline_coeffs = []
for i in range(1, len(x_nodes)):
    idx = i
    prev = i - 1
    hi = h_step
    ai = y_nodes[idx]
    ci = c_coeffs[idx]
    c_prev = c_coeffs[prev]
    di = (ci - c_prev) / hi
    bi = (hi * ci) / 2 - (hi**2 * di) / 6 + (y_nodes[idx] - y_nodes[prev]) / hi
    spline_coeffs.append({'range': (x_nodes[prev], x_nodes[idx]), 'xr': x_nodes[idx], 'a': ai, 'b': bi, 'c': ci, 'd': di})

def eval_spline(x, coeffs):
    for interval in coeffs:
        x_start, x_end = interval['range']
        if x_start <= x <= x_end + 1e-9: # +epsilon для точності на краю
            dx = x - interval['xr']
            return interval['a'] + interval['b']*dx + (interval['c']/2)*dx**2 + (interval['d']/6)*dx**3
    return 0.0

def eval_linear_explicit(x, xn, yn):
    for i in range(len(xn) - 1):
        if xn[i] <= x <= xn[i+1] + 1e-9:
            # Явний розрахунок за формулою
            return (yn[i] * (xn[i+1] - x) + yn[i+1] * (x - xn[i])) / (xn[i+1] - xn[i])
    return 0.0

def eval_quadratic_explicit(x, xn, yn, h):
    term1 = yn[0] * (x - xn[1]) * (x - xn[2]) / (2 * h**2)
    term2 = yn[1] * (x - xn[0]) * (x - xn[2]) / (h**2)
    term3 = yn[2] * (x - xn[0]) * (x - xn[1]) / (2 * h**2)
    return term1 - term2 + term3

x_plot = np.linspace(1, 5, 200)
y_true = f(x_plot)

y_spline = [eval_spline(x, spline_coeffs) for x in x_plot]
y_linear = [eval_linear_explicit(x, x_nodes, y_nodes) for x in x_plot]
y_quad   = [eval_quadratic_explicit(x, x_nodes, y_nodes, h_step) for x in x_plot]

err_spline = np.abs(y_true - y_spline)
err_linear = np.abs(y_true - y_linear)
err_quad   = np.abs(y_true - y_quad)

print(f"--- Максимальні похибки ---")
print(f"Сплайн:          {np.max(err_spline):.4f}")
print(f"Кусково-лінійна: {np.max(err_linear):.4f}")
print(f"Квадратична:     {np.max(err_quad):.4f}")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x) (Точна)', alpha=0.3)
plt.plot(x_plot, y_spline, 'r--', label='Кубічний сплайн')
plt.plot(x_plot, y_linear, 'g:', label='Кусково-лінійна')
plt.plot(x_plot, y_quad, 'b-.', label='Квадратична (формула з методички)')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Вузли')
plt.legend(); plt.grid(True)
plt.title('Порівняння методів інтерполяції')

plt.subplot(1, 2, 2)
plt.plot(x_plot, err_spline, 'r', label='Похибка сплайна')
plt.plot(x_plot, err_linear, 'g', label='Похибка лінійної')
plt.plot(x_plot, err_quad, 'b', label='Похибка квадратичної')
plt.yscale('log'); plt.legend(); plt.grid(True)
plt.title('Абсолютна похибка')
plt.tight_layout()
plt.show()