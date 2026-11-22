import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import os

# =====================================================================
# パラメータ
# =====================================================================
tf = 1.0     # 動作時間 [s]
dt = 0.005
N = int(round(tf / dt))   # 区間の数（節点は N + 1 個）

# モデルのパラメータ
m1 = 0.06      # 車輪2輪の合計質量 [kg]（暫定値。未測定）
m2 = 0.288     # 車体質量 [kg]（実測：電池込み、ホイール除外）
I1 = 4.8e-5    # 車輪2輪の合計慣性 [kg m^2]（暫定値。未測定）
I2 = 2.2e-4    # 車体の重心まわり慣性 [kg m^2]（振り子法）
s = 0.0165     # 車軸から車体重心まで [m]（実測）
r = 0.08 / 2   # 車輪半径 [m]（実測：直径 8 cm）
g = 9.81

# 状態 X = [x, theta, dx, dtheta]（theta は水平から反時計回り、pi/2 が直立）
x0 = np.array([0.0, 0.0, 0.0, 0.0])           # 前に寝た状態
xf = np.array([0.0, np.pi / 2, 0.0, 0.0])     # 直立して静止（x は自由）

# モータのパラメータ
Ra = 3.75      # 巻線抵抗 [Ohm]
Kt = 0.194     # トルク定数 [N m/A]
Ke = 0.264     # 逆起電力定数 [V s/rad]

# 実機の制約（上限の8割に設定）
V_batt = 1.2 * 6              # 電池電圧 [V]（1.2 V の電池 6 本）
V_max = 0.8 * V_batt          # モータにかけられる電圧の上限 [V]
mu = 1.0                      # 床とタイヤの摩擦係数（仮の値。実測が必要）
a_max = 0.8 * mu * g          # 車速の加速度の上限 [m/s^2]（スリップしない目安）


# =====================================================================
# モデル
# =====================================================================
def f(X, u):
    """状態方程式  M(theta) [ddx, ddtheta] = b を解いて dX/dt を返す"""
    th, dth = X[1], X[3]
    M11 = m1 + m2 + I1 / r**2
    M12 = -m2 * s * ca.sin(th)
    M22 = I2 + m2 * s**2
    b1 = m2 * s * ca.cos(th) * dth**2 + u / r
    b2 = -m2 * g * s * ca.cos(th) + u
    det = M11 * M22 - M12**2
    ddx = (M22 * b1 - M12 * b2) / det
    ddth = (M11 * b2 - M12 * b1) / det
    return ca.vertcat(X[2], X[3], ddx, ddth)


def voltage(X, u):
    """モータ1個に必要な電圧（u は2輪合計なので1輪分は u/2）"""
    omega_rel = X[2] / r + X[3]          # 車体に対する車輪の回転速度
    return Ra / Kt * (u / 2) + Ke * omega_rel


def rk4(X, u):
    k1 = f(X, u)
    k2 = f(X + dt / 2 * k1, u)
    k3 = f(X + dt / 2 * k2, u)
    k4 = f(X + dt * k3, u)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# =====================================================================
# 直接法（多重シューティング、CasADi＋IPOPT）
# =====================================================================
def optimize():
    opti = ca.Opti()
    X = opti.variable(4, N + 1)          # 各節点の状態
    U = opti.variable(1, N)              # 各区間のトルク（区間内で一定）

    J = 0
    for k in range(N):
        opti.subject_to(X[:, k + 1] == rk4(X[:, k], U[k]))                     # 区間のつながり
        opti.subject_to(opti.bounded(-V_max, voltage(X[:, k], U[k]), V_max))   # 電圧
        opti.subject_to(opti.bounded(-a_max, f(X[:, k], U[k])[2], a_max))      # 加速度（スリップ）
        J += U[k]**2 * dt                                                      # 入力（消費エネルギー）

    opti.subject_to(opti.bounded(0, X[1, :], np.pi))    # 床を突き抜けない
    opti.subject_to(X[:, 0] == x0)                      # 初期状態
    opti.subject_to(X[1:, N] == xf[1:])                 # 終端状態

    opti.minimize(J)
    opti.set_initial(X[1, :], np.linspace(x0[1], xf[1], N + 1))    # 初期値：theta を直線で
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "max_iter": 3000})
    sol = opti.solve()
    return sol.value(X).T, sol.value(U), sol.value(J)


def main():
    x, u, J = optimize()
    t = np.linspace(0, tf, N + 1)

    print(f"J = {J:.6f}")
    print(f"終端状態 x = {x[-1, 0]:.3f} m, theta = {np.degrees(x[-1, 1]):.2f} deg, "
          f"dx = {x[-1, 2]:.3f} m/s, dtheta = {x[-1, 3]:.3f} rad/s")
    print(f"theta の範囲 {np.degrees(x[:, 1].min()):.1f} ～ {np.degrees(x[:, 1].max()):.1f} deg")
    print(f"最大トルク |tau| = {np.abs(u).max():.3f} N m")

    # 実機の制約を守れているか確認
    V = np.array([float(voltage(x[k], u[k])) for k in range(N)])
    ddx = np.array([float(f(x[k], u[k])[2]) for k in range(N)])
    print(f"最大電圧   |V|     = {np.abs(V).max():.2f} V      （上限 {V_max:.2f} V）")
    print(f"最大加速度 |ddx|   = {np.abs(ddx).max():.2f} m/s^2  （上限 {a_max:.2f} m/s^2）")

    # プログラムと同じフォルダに保存する（最後の行の tau は使わないので 0）
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, "swingup_direct.csv")
    data = np.column_stack([t, x, np.append(u, 0.0)])
    np.savetxt(path, data, delimiter=",",
               header="t,x,theta,dx,dtheta,tau", comments="", fmt="%.6f")
    print("保存しました:", path)

    fig, ax = plt.subplots(3, 2, figsize=(9, 8))
    ax[0, 0].step(t[:-1], u, where="post"); ax[0, 0].set_title("tau [N m]")
    ax[0, 1].plot(t, x[:, 0]); ax[0, 1].set_title("x [m]")
    ax[1, 0].plot(t, x[:, 1]); ax[1, 0].set_title("theta [rad]")
    ax[1, 0].axhline(np.pi / 2, linestyle="--", color="gray")   # 直立
    ax[1, 1].plot(t, x[:, 2]); ax[1, 1].set_title("dx [m/s]")
    ax[2, 0].plot(t, x[:, 3]); ax[2, 0].set_title("dtheta [rad/s]")
    ax[2, 1].plot(t[:-1], V); ax[2, 1].set_title("V [V]")
    for v in (V_max, -V_max):
        ax[2, 1].axhline(v, linestyle="--", color="gray")       # 電圧の上限
    for a in ax.flat:
        a.set_xlabel("t [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(folder, "swingup_direct.png"))
    plt.show()


if __name__ == "__main__":
    main()
