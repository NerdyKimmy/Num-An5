import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2 * x ** 6 + 3 * x ** 5 + 4 * x ** 2 - 1


def df2(x):
    return 60 * x ** 4 + 60 * x ** 3 + 8


x_nodes = np.array([1.0, 3.0, 5.0])
y_nodes = f(x_nodes)
h = np.diff(x_nodes)


c0 = df2(x_nodes[0])
c2 = df2(x_nodes[-1])

rhs = 6 * ((y_nodes[2] - y_nodes[1]) / h[1] - (y_nodes[1] - y_nodes[0]) / h[0])
lhs_known = h[0] * c0 + h[1] * c2
c1 = (rhs - lhs_known) / (2 * (h[0] + h[1]))

c_coeffs = [c0, c1, c2]

spline_coeffs = []
for i in range(1, len(x_nodes)):
    idx = i
    prev = i - 1

    hi = h[prev]
    ai = y_nodes[idx]
    ci = c_coeffs[idx]
    c_prev = c_coeffs[prev]

    di = (ci - c_prev) / hi
    bi = (hi * ci) / 2 - (hi ** 2 * di) / 6 + (y_nodes[idx] - y_nodes[prev]) / hi

    spline_coeffs.append({
        'range': (x_nodes[prev], x_nodes[idx]),
        'xr': x_nodes[idx],
        'a': ai,
        'b': bi,
        'c': ci,
        'd': di
    })


def eval_spline(x, coeffs):
    for interval in coeffs:
        x_start, x_end = interval['range']
        if x_start <= x <= x_end:
            dx = x - interval['xr']  # (x - xi)
            return interval['a'] + interval['b'] * dx + (interval['c'] / 2) * dx ** 2 + (interval['d'] / 6) * dx ** 3
    return None


def eval_quadratic(x, xn, yn):
    # Розділені різниці
    f01 = (yn[1] - yn[0]) / (xn[1] - xn[0])
    f12 = (yn[2] - yn[1]) / (xn[2] - xn[1])
    f012 = (f12 - f01) / (xn[2] - xn[0])

    return yn[0] + f01 * (x - xn[0]) + f012 * (x - xn[0]) * (x - xn[1])


x_plot = np.linspace(1, 5, 200)
y_true = f(x_plot)
y_spline = [eval_spline(x, spline_coeffs) for x in x_plot]
y_linear = np.interp(x_plot, x_nodes, y_nodes)  # Кусково-лінійна (стандартна ф-ція)
y_quad = [eval_quadratic(x, x_nodes, y_nodes) for x in x_plot]

err_spline = np.abs(y_true - y_spline)
err_linear = np.abs(y_true - y_linear)
err_quad = np.abs(y_true - y_quad)

print(f"--- Результати розрахунку коефіцієнтів ---")
for i, coef in enumerate(spline_coeffs):
    print(f"Проміжок {i + 1} [{coef['range'][0]}..{coef['range'][1]}]:")
    print(f"  a = {coef['a']:.4f}")
    print(f"  b = {coef['b']:.4f}")
    print(f"  c = {coef['c']:.4f}")
    print(f"  d = {coef['d']:.4f}")

print(f"\n--- Максимальні похибки ---")
print(f"Сплайн (модифікований): {np.max(err_spline):.4f}")
print(f"Кусково-лінійна:        {np.max(err_linear):.4f}")
print(f"Квадратична:            {np.max(err_quad):.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x) (Точна)', alpha=0.3)
plt.plot(x_plot, y_spline, 'r--', label='Кубічний сплайн')
plt.plot(x_plot, y_linear, 'g:', label='Кусково-лінійна')
plt.plot(x_plot, y_quad, 'b-.', label='Квадратична (поліном Ньютона)')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Вузли')
plt.title('Порівняння методів інтерполяції')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_plot, err_spline, 'r', label='Похибка сплайна')
plt.plot(x_plot, err_linear, 'g', label='Похибка лінійної')
plt.plot(x_plot, err_quad, 'b', label='Похибка квадратичної')
plt.title('Абсолютна похибка |f(x) - P(x)|')
plt.xlabel('x')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()