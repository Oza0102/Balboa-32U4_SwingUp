"""
=========================================================================================
                     「物体持ち上げ動作における操作力最小化問題」
                              solved by 最急降下法

       物体を持ち上げる際の操作力を最小化する最適制御問題を、最急降下法（間接法）で解く。
    ハミルトニアンの偏微分はSymPyで自動的に求め、随伴方程式を用いて評価関数の勾配を計算する。
=========================================================================================
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ===== パラメータ =====
tf = 1      # 終端時刻
dt = 0.01   # 時間刻み幅
N = int(round(tf / dt)) + 1 # 時間ステップ数

m = 1.0     # 質量
g = 9.81    # 重力加速度
h = 1.0     # 目標位置
w = 100     # 終端コストの重み

alpha = 0.01
tol = 1e-4

# ===== SymPy で H と φ を定義し、偏微分を自動で求める =====
# 記号（文字）を用意する
x1, x2, u_s, l1, l2 = sp.symbols("x1 x2 u lambda1 lambda2")
x_s = [x1, x2]

# 状態方程式の右辺 f、評価関数の被積分項 L、終端コスト φ
f_s = [x2, u_s / m - g]
L_s = u_s**2 / 2
phi_s = w / 2 * ((x1 - h)**2 + x2**2)

# ハミルトニアン H = L + λ1*f1 + λ2*f2
H_s = L_s + l1 * f_s[0] + l2 * f_s[1]

# 偏微分を SymPy 自動で求める
minus_dHdx_s = [-sp.diff(H_s, xi) for xi in x_s]   # 随伴方程式の右辺 -∂H/∂x
dHdu_s = sp.diff(H_s, u_s)                         # ∂H/∂u
dphidx_s = [sp.diff(phi_s, xi) for xi in x_s]      # λ(tf) = ∂φ/∂x

print("H      =", H_s)
print("-dH/dx =", minus_dHdx_s)
print("dH/du  =", dHdu_s)
print("dphi/dx=", dphidx_s)

# 記号の式を、NumPy で計算できる関数に変換する
f_func = sp.lambdify((x1, x2, u_s), f_s, "numpy")
minus_dHdx_func = sp.lambdify((x1, x2, u_s, l1, l2), minus_dHdx_s, "numpy")
dHdu_func = sp.lambdify((x1, x2, u_s, l1, l2), dHdu_s, "numpy")
dphidx_func = sp.lambdify((x1, x2), dphidx_s, "numpy")
phi_func = sp.lambdify((x1, x2), phi_s, "numpy")



def f(x, u):
    return np.array(f_func(x[0], x[1], u), dtype=float)


def solve_x(x0, u):
    x = np.zeros((N, 2))
    dx = np.zeros((N, 2))
    x[0] = x0
    for k in range(N - 1):
        k1 = f(x[k], u[k])
        k2 = f(x[k] + dt / 2 * k1, u[k])
        k3 = f(x[k] + dt / 2 * k2, u[k])
        k4 = f(x[k] + dt * k3, u[k])
        dx[k] = k1
        x[k + 1] = x[k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    dx[N - 1] = f(x[N - 1], u[N - 1])
    return x, dx


def dH_dx(lam, x, u):
    """随伴方程式の右辺 -∂H/∂x を返す"""
    return np.array(minus_dHdx_func(x[0], x[1], u, lam[0], lam[1]), dtype=float)


def solve_lam(x, u):
    lam = np.zeros((N, 2))
    dlam = np.zeros((N, 2))
    lam[N - 1] = dphidx_func(x[N - 1, 0], x[N - 1, 1])   # λ(tf) = ∂φ/∂x
    for k in range(N - 1, 0, -1):
        xm = (x[k] + x[k - 1]) / 2
        k1 = dH_dx(lam[k], x[k], u[k - 1])
        k2 = dH_dx(lam[k] - dt / 2 * k1, xm, u[k - 1])
        k3 = dH_dx(lam[k] - dt / 2 * k2, xm, u[k - 1])
        k4 = dH_dx(lam[k] - dt * k3, x[k - 1], u[k - 1])
        dlam[k] = k1
        lam[k - 1] = lam[k] - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    dlam[0] = dH_dx(lam[0], x[0], u[0])
    return lam, dlam


def solve_dHdu(lam, x, u):
    return dHdu_func(x[:, 0], x[:, 1], u, lam[:, 0], lam[:, 1])


def cost(x, u):
    phi = phi_func(x[N - 1, 0], x[N - 1, 1])
    L = 0.5 * np.sum(u[:-1]**2) * dt
    return phi + L


def main():
    x0 = np.array([0.0, 0.0])
    u = np.zeros(N)

    max_iter = 10000
    for i in range(max_iter):
        x, dx = solve_x(x0, u)
        lam, dlam = solve_lam(x, u)
        dHdu = solve_dHdu(lam, x, u)
        dHdu_norm = np.sqrt(np.sum(dHdu**2) * dt)
        J = cost(x, u)
        if dHdu_norm < tol:
            print(f"Converged after {i} iterations.")
            break
        u = u - alpha * dHdu

    x, dx = solve_x(x0, u)
    print(f"J = {cost(x, u):.6f}   x1(tf) = {x[-1, 0]:.4f}   x2(tf) = {x[-1, 1]:.4f}")

    t = np.linspace(0, tf, N)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(t, u);       ax[0].set_title("u")
    ax[1].plot(t, x[:, 0]); ax[1].set_title("x1")
    ax[2].plot(t, x[:, 1]); ax[2].set_title("x2")
    for a in ax:
        a.set_xlabel("t [s]")
    fig.suptitle("SymPy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
