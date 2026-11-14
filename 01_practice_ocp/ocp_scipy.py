import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ===== パラメータ =====
tf = 1
dt = 0.01
N = int(round(tf / dt)) + 1

m = 1.0
g = 9.81
h = 1.0
w = 100

# True にすると、ペナルティではなく x1(tf)=h, x2(tf)=0 を厳密に満たす問題を解く
hard_constraint = False


# ===== 2点境界値問題として解く =====
# 状態と随伴変数をまとめて y = [x1, x2, λ1, λ2] とする。
# ∂H/∂u = u + λ2/m = 0 より、最適な u は u = -λ2/m で決まる。

def u_opt(y):
    return -y[3] / m


def ode(t, y):
    """y = [x1, x2, λ1, λ2] の微分を返す（t の各点ぶんまとめて計算する）"""
    u = u_opt(y)
    dx1 = y[1]
    dx2 = u / m - g
    dl1 = np.zeros_like(y[0])   # -∂H/∂x1 = 0
    dl2 = -y[2]                 # -∂H/∂x2 = -λ1
    return np.vstack([dx1, dx2, dl1, dl2])


def bc(ya, yb):
    """境界条件。ya は t=0、yb は t=tf での y。各式が 0 になるように解かれる"""
    if hard_constraint:
        return np.array([
            ya[0] - 0.0,          # x1(0) = 0
            ya[1] - 0.0,          # x2(0) = 0
            yb[0] - h,            # x1(tf) = h
            yb[1] - 0.0,          # x2(tf) = 0
        ])
    else:
        return np.array([
            ya[0] - 0.0,                  # x1(0) = 0
            ya[1] - 0.0,                  # x2(0) = 0
            yb[2] - w * (yb[0] - h),      # λ1(tf) = w(x1(tf) - h)
            yb[3] - w * yb[1],            # λ2(tf) = w x2(tf)
        ])


def main():
    t = np.linspace(0, tf, N)
    y_guess = np.zeros((4, N))          # 初期推定（すべて 0）

    sol = solve_bvp(ode, bc, t, y_guess, tol=1e-6)
    print("solve_bvp:", sol.message)

    y = sol.sol(t)                      # 解を時刻 t の点で取り出す
    x1, x2 = y[0], y[1]
    u = u_opt(y)

    # u は滑らかな連続関数として求まるので、積分は台形公式で計算する
    J = 0.5 * np.sum((u[:-1]**2 + u[1:]**2) / 2) * dt
    if not hard_constraint:
        J += 0.5 * w * ((x1[-1] - h)**2 + x2[-1]**2)
    print(f"J = {J:.6f}   x1(tf) = {x1[-1]:.4f}   x2(tf) = {x2[-1]:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(t, u);  ax[0].set_title("u")
    ax[1].plot(t, x1); ax[1].set_title("x1")
    ax[2].plot(t, x2); ax[2].set_title("x2")
    for a in ax:
        a.set_xlabel("t [s]")
    fig.suptitle("SciPy solve_bvp")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
