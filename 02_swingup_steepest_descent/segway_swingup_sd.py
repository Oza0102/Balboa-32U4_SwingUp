import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os

# =====================================================================
# パラメータ
# =====================================================================
tf = 0.5     # 動作時間 [s]
dt = 0.005
N = int(round(tf / dt)) + 1

# モデルのパラメータ
m1 = 0.06      # 車輪2輪の合計質量 [kg]（暫定値。未測定）
m2 = 0.288     # 車体質量 [kg]（実測：電池込み、ホイール除外）
I1 = 4.8e-5    # 車輪2輪の合計慣性 [kg m^2]（暫定値。未測定）
I2 = 1.0e-3    # 車体の重心まわり慣性 [kg m^2]（暫定値。未測定）
s = 0.0165     # 車軸から車体重心まで [m]（実測）
r = 0.08 / 2   # 車輪半径 [m]（実測：直径 8 cm）
g = 9.81

# 状態 X = [x, theta, dx, dtheta]（theta は水平から反時計回り、pi/2 が直立）
x0 = np.array([0.0, 0.0, 0.0, 0.0])               # 前に寝た状態
xf = np.array([0.0, np.pi / 2, 0.0, 0.0])     # 直立して静止

# モータのパラメータ
Ra = 3.75      # 巻線抵抗 [Ohm]
Kt = 0.194     # トルク定数 [N m/A]
Ke = 0.264     # 逆起電力定数 [V s/rad]

# 実機の制約（上限の8割に設定）
V_batt = 1.2 * 6               # 電池電圧 [V]（1.2 V の電池 6 本）
V_max = 0.8 * V_batt          # モータにかけられる電圧の上限 [V]
mu = 0.5                      # 床とタイヤの摩擦係数（仮の値。実測が必要）
a_max = 0.8 * mu * g          # 車速の加速度の上限 [m/s^2]（スリップしない目安）

# 評価関数の重み
rho = 1.0                                   # 動作時間の重み（今回は使わない）
R = 1.0                                     # 入力（消費エネルギー）
S = np.array([0.0, 100.0, 10.0, 10.0])      # 終端ペナルティ（[x, theta, dx, dtheta] の重み）
W_V = 100.0                                 # 電圧の上限を超えたときのペナルティ
W_a = 0.0                                   # 加速度の上限を超えたときのペナルティ（今回は使わない）
W_floor = 1e4                               # 床を突き抜けたとき（theta < 0 または theta > pi）のペナルティ

# 最急降下法の設定
alpha0 = 1e-6         # ステップ幅の初期値
tol = 1e-5            # 勾配ノルムが初回の tol 倍より小さくなったら収束
max_iter = 10000


# =====================================================================
# SymPy でハミルトニアンを作り、偏微分を自動で求める
# =====================================================================
xs, ths, dxs, dths = sp.symbols("x theta dx dtheta")
X_s = [xs, ths, dxs, dths]
u_s = sp.symbols("tau")
lam_s = sp.symbols("lambda1:5")           # 状態が4つなので λ も4つ

# 状態方程式  M(theta) [ddx, ddtheta] = b
M_s = sp.Matrix([
    [m1 + m2 + I1 / r**2, -m2 * s * sp.sin(ths)],
    [-m2 * s * sp.sin(ths), I2 + m2 * s**2],
])
b_s = sp.Matrix([
    m2 * s * sp.cos(ths) * dths**2 + u_s / r,
    -m2 * g * s * sp.cos(ths) + u_s,
])
ddq_s = M_s.LUsolve(b_s)
f_s = [dxs, dths, ddq_s[0], ddq_s[1]]


# モータ1個に必要な電圧Vと、車速の加速度ddx
omega_rel_s = dxs / r + dths                              # 車体に対する車輪の回転速度
V_s = Ra / Kt * (u_s / 2) + Ke * omega_rel_s               # u_sは2輪合計なので1輪分はu_s/2
ddx_s = ddq_s[0]

# 評価関数の制約条件を表す関数
def over_limit(z, z_max):
    """|z| が z_max を超えた分の2乗（超えていなければ 0）"""
    return sp.Piecewise(((z - z_max)**2, z > z_max),
                        ((z + z_max)**2, z < -z_max),
                        (0, True))


# 評価関数
P_V_s = W_V / 2 * over_limit(V_s, V_max)                   # 電圧（トルクと回転速度）の制約
P_a_s = W_a / 2 * over_limit(ddx_s, a_max)                 # 加速度（スリップ）の制約
P_floor_s = W_floor / 2 * sp.Piecewise((ths**2, ths < 0),((ths - sp.pi)**2, ths > sp.pi),(0, True)) # 床を突き抜けたときの制約
L_s = rho + R / 2 * u_s**2 + P_V_s + P_a_s + P_floor_s
phi_s = sum(S[i] / 2 * (X_s[i] - xf[i])**2 for i in range(4)) # 終端ペナルティ

# ハミルトニアンと偏微分
H_s = L_s + sum(lam_s[i] * f_s[i] for i in range(4))        # H = L + λ^T f
minus_dHdx_s = [-sp.diff(H_s, Xi) for Xi in X_s]            # 随伴方程式の右辺 -∂H/∂x
dHdu_s = sp.diff(H_s, u_s)                                  # ∂H/∂u
dphidx_s = [sp.diff(phi_s, Xi) for Xi in X_s]               # λ(tf) = ∂φ/∂x

# NumPy で計算できる関数に変換（cse=True で共通部分をまとめて計算を速くする）
f_func = sp.lambdify([X_s, u_s], f_s, "math", cse=True)
minus_dHdx_func = sp.lambdify([X_s, u_s, lam_s], minus_dHdx_s, "math", cse=True)
dHdu_func = sp.lambdify([X_s, u_s, lam_s], dHdu_s, "numpy", cse=True)
dphidx_func = sp.lambdify([X_s], dphidx_s, "numpy")
phi_func = sp.lambdify([X_s], phi_s, "numpy")
L_func = sp.lambdify([X_s, u_s], L_s, "numpy", cse=True)          # 評価関数の被積分項
V_func = sp.lambdify([X_s, u_s], V_s, "numpy")                    # 結果の確認用
ddx_func = sp.lambdify([X_s, u_s], ddx_s, "numpy", cse=True)      # 結果の確認用


# =====================================================================
# ここから下は ocp_sympy.py とほぼ同じ構成
# =====================================================================
# 状態方程式を初期条件から順時間方向に解く
def f(x, u):
    return np.array(f_func(x, u), dtype=float)


def solve_x(x0, u):
    x = np.zeros((N, 4))
    dx = np.zeros((N, 4))
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


# dλ/dt = -∂H/∂x を逆時間方向に解く
def dH_dx(lam, x, u):
    """随伴方程式の右辺 -∂H/∂x を返す"""
    return np.array(minus_dHdx_func(x, u, lam), dtype=float)


def solve_lam(x, u):
    lam = np.zeros((N, 4))
    dlam = np.zeros((N, 4))
    lam[N - 1] = dphidx_func(x[N - 1])
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
    return dHdu_func(x.T, u, lam.T)


# 評価関数 J を計算
def cost(x, u):
    phi = phi_func(x[N - 1])
    L = L_func(x[:-1].T, u[:-1])          # H で使っているのと同じ L を各時刻で計算する
    return phi + np.sum(L) * dt


def cost_of_u(u):
    """u から J を計算する。計算が発散したら無限大を返す"""
    try:
        x, dx = solve_x(x0, u)
        J = cost(x, u)
    except (ValueError, OverflowError):
        return np.inf
    if not np.isfinite(J):
        return np.inf
    return J


# =====================================================================
# 最急降下法
# =====================================================================
def optimize(u):
    alpha = alpha0
    for i in range(max_iter):
        x, dx = solve_x(x0, u)
        lam, dlam = solve_lam(x, u)
        dHdu = solve_dHdu(lam, x, u)
        dHdu_norm = np.sqrt(np.sum(dHdu**2) * dt)
        J = cost(x, u)
        if i == 0:
            norm0 = dHdu_norm
        print(f"反復 {i:5d}  J = {J:.6f}  norm = {dHdu_norm:.3e}  alpha = {alpha:.2e}")

        if dHdu_norm < tol * norm0:
            print(f"収束しました（反復回数 {i}）")
            return u, True

        # J が減るまでステップ幅を半分にする
        for _ in range(60):
            u_new = u - alpha * dHdu
            if cost_of_u(u_new) < J:
                break
            alpha = alpha / 2
        else:
            # 60回半分にしても J が減らない場合
            print("収束しませんでした（J を減らせるステップ幅が見つかりません）")
            return u, False
        u = u_new
        alpha = alpha * 2          # 次の反復では少し大きいステップから試す

    print(f"収束しませんでした（反復回数の上限 {max_iter} 回に達しました）")
    return u, False


def main():
    u = np.zeros(N)
    u, converged = optimize(u)

    x, dx = solve_x(x0, u)
    t = np.linspace(0, tf, N)

    if not converged:
        print("※ 以下は収束していない途中の結果です")
    print(f"J = {cost(x, u):.6f}")
    print(f"終端状態 x = {x[-1, 0]:.3f} m, theta = {np.degrees(x[-1, 1]):.2f} deg, "
          f"dx = {x[-1, 2]:.3f} m/s, dtheta = {x[-1, 3]:.3f} rad/s")
    print(f"theta の範囲 {np.degrees(x[:, 1].min()):.1f} ～ {np.degrees(x[:, 1].max()):.1f} deg"
          f"（0 未満と 180 超は床を突き抜けている）")
    print(f"最大トルク |tau| = {np.abs(u[:-1]).max():.3f} N m")

    # 実機の制約を守れているか確認
    V = V_func(x[:-1].T, u[:-1])
    ddx = ddx_func(x[:-1].T, u[:-1])
    omega_rel = x[:, 2] / r + x[:, 3]
    print(f"最大電圧   |V|     = {np.abs(V).max():.2f} V      （上限 {V_max:.2f} V）")
    print(f"最大加速度 |ddx|   = {np.abs(ddx).max():.2f} m/s^2  （上限 {a_max:.2f} m/s^2）")
    print(f"最大回転速度 |omega| = {np.abs(omega_rel).max():.1f} rad/s"
          f"（電圧の上限だけで回せる速さ {V_max / Ke:.1f} rad/s）")

    # プログラムと同じフォルダに保存
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, "swingup_sd.csv")
    data = np.column_stack([t, x, u])
    np.savetxt(path, data, delimiter=",",
               header="t,x,theta,dx,dtheta,tau", comments="", fmt="%.6f")
    print("保存しました:", path)

    fig2, ax2 = plt.subplots()
    ax2.plot(t[:-1], u[:-1])
    ax2.set_title("tau [N m]")
    ax2.set_xlabel("t [s]")

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(t, x[:, 0]); ax[0, 0].set_title("x [m]")
    ax[0, 1].plot(t, x[:, 1]); ax[0, 1].set_title("theta [rad]")
    ax[0, 1].axhline(np.pi / 2, linestyle="--", color="gray")   # 直立
    ax[1, 0].plot(t, x[:, 2]); ax[1, 0].set_title("dx [m/s]")
    ax[1, 1].plot(t, x[:, 3]); ax[1, 1].set_title("dtheta [rad/s]")
    for a in ax.flat:
        a.set_xlabel("t [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()