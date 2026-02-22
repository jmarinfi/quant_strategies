from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import statsmodels.api as sm
import optuna
import matplotlib.pyplot as plt
import ccxt
import requests
import json


def bollinger_pair_trading(
    price_y: pd.Series, price_x: pd.Series, lookback: int, entry_z: float, exit_z: float
) -> pd.DataFrame:
    # Cálculo del hedge ratio
    model = sm.OLS(price_y, sm.add_constant(price_x)).fit()
    hedge_ratio = model.params.iloc[1]

    # Cálculo del spread
    spread = price_y - (hedge_ratio * price_x)

    # Cálculo de las bandas de bollinger y z-score
    moving_avg = spread.rolling(window=lookback).mean()
    moving_std = spread.rolling(window=lookback).std()
    z_score = (spread - moving_avg) / moving_std

    # Generación de señales
    df = pd.DataFrame(
        {"price_y": price_y, "price_x": price_x, "spread": spread, "z_score": z_score}
    )

    df["longs_entry"] = df["z_score"] < -entry_z
    df["longs_exit"] = df["z_score"] >= -exit_z
    df["shorts_entry"] = df["z_score"] > entry_z
    df["shorts_exit"] = df["z_score"] <= exit_z

    # Cálculo de Unidades en Largo
    df["units_long"] = np.nan
    df.loc[df["longs_entry"], "units_long"] = 1
    df.loc[df["longs_exit"], "units_long"] = 0
    df["units_long"] = df["units_long"].ffill().fillna(0)

    # Cálculo de Unidades en Corto
    df["units_short"] = np.nan
    df.loc[df["shorts_entry"], "units_short"] = -1
    df.loc[df["shorts_exit"], "units_short"] = 0
    df["units_short"] = df["units_short"].ffill().fillna(0)

    # Posición neta total y almacenamiento del hedge ratio
    df["net_units"] = df["units_long"] + df["units_short"]
    df["hedge_ratio"] = hedge_ratio

    return df


def calcular_backtest(
    df: pd.DataFrame, capital_inicial: float = 10000.0, fee_rate: float = 0.001
) -> pd.DataFrame:
    df = df.copy()

    # Escalar posiciones al capital inicial para evitar apalancamiento infinito
    # El coste de 1 unidad de spread es el precio de Y + el precio de X * hedge_ratio
    coste_unitario_spread = df["price_y"] + (df["hedge_ratio"] * df["price_x"])
    unidades_ajustadas = capital_inicial / coste_unitario_spread

    # 1. Posiciones físicas escaladas (Fracciones de BTC y SOL)
    df["pos_Y"] = df["net_units"] * unidades_ajustadas
    df["pos_X"] = -df["net_units"] * df["hedge_ratio"] * unidades_ajustadas

    # 2. Diferencia de precios absolutos
    df["delta_price_y"] = df["price_y"].diff()
    df["delta_price_x"] = df["price_x"].diff()

    # 3. PnL Bruto (Uso de shift(1) para evitar sesgo de anticipación)
    df["pnl_bruto"] = (df["pos_Y"].shift(1) * df["delta_price_y"]) + (
        df["pos_X"].shift(1) * df["delta_price_x"]
    )
    df["pnl_bruto"] = df["pnl_bruto"].fillna(0)

    # 4. Cálculo de comisiones por transacciones
    df["trade_Y"] = df["pos_Y"].diff().fillna(0)
    df["trade_X"] = df["pos_X"].diff().fillna(0)

    df["comisiones"] = (df["trade_Y"].abs() * df["price_y"] * fee_rate) + (
        df["trade_X"].abs() * df["price_x"] * fee_rate
    )

    # 5. Curva de capital y retornos (Corregido para evitar errores matemáticos con balances negativos)
    df["pnl_neto"] = df["pnl_bruto"] - df["comisiones"]
    df["equity"] = capital_inicial + df["pnl_neto"].cumsum()

    # Retorno simple en base al capital inicial constante (evita distorsión de pct_change)
    df["retorno"] = df["pnl_neto"] / capital_inicial

    return df


def calcular_metricas(df: pd.DataFrame, periodos_por_ano: int) -> dict:
    equity = df["equity"]
    returns = df["retorno"]

    # Retorno Total
    retorno_total = (equity.iloc[-1] / equity.iloc[0]) - 1

    # Sharpe Ratio
    volatilidad = returns.std()
    sharpe_ratio = (
        np.sqrt(periodos_por_ano) * (returns.mean() / volatilidad)
        if volatilidad > 0
        else 0
    )

    # Maximum Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "Capital Final": round(equity.iloc[-1], 2),
        "Retorno Total (%)": round(retorno_total * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
    }


def run_optimization(price_y: pd.Series, price_x: pd.Series, exit_z: float):
    def objective(trial: optuna.Trial):
        lookback = trial.suggest_int(
            "lookback", 10, 200
        )  # Límite inferior elevado para evitar exceso de comisiones
        entry_z = trial.suggest_float("entry_z", 1.0, 3.5)

        df_signals = bollinger_pair_trading(price_y, price_x, lookback, entry_z, exit_z)
        df_backtest = calcular_backtest(df_signals)
        metricas = calcular_metricas(df_backtest, periodos_por_ano=35040)

        # Penalización severa si la estrategia pierde dinero
        if metricas["Capital Final"] < 10000.0:
            return -999.0

        sharpe = metricas["Sharpe Ratio"]
        if pd.isna(sharpe) or sharpe == float("inf"):
            return -999.0

        return float(sharpe)

    print("Iniciando optimización de hiperparámetros con Optuna...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Limpiar consola

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    print("\n¡Optimización completada!")
    print(f"Mejor Sharpe Ratio: {study.best_value:.4f}")
    print(
        f"Mejores parámetros:\n - Lookback: {study.best_params['lookback']}\n - Entry Z: {study.best_params['entry_z']}"
    )

    return study.best_params


def plot_results(df: pd.DataFrame, entry_z: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Gráfico 1: Curva de Capital
    ax1.plot(df.index, df["equity"], label="Equity", color="blue", linewidth=1.5)
    ax1.set_title("Curva de Capital (Equity)")
    ax1.set_ylabel("Capital (USD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfico 2: Z-Score del Spread y Señales Operativas
    ax2.plot(
        df.index, df["z_score"], label="Z-Score del Spread", color="gray", alpha=0.7
    )
    ax2.axhline(entry_z, color="red", linestyle="--", label="Umbral Entrada Corto")
    ax2.axhline(-entry_z, color="green", linestyle="--", label="Umbral Entrada Largo")
    ax2.axhline(0, color="black", linestyle=":", label="Umbral Salida (Media)")

    # Marcadores de señales
    longs = df[df["longs_entry"]]
    shorts = df[df["shorts_entry"]]
    ax2.scatter(
        longs.index,
        longs["z_score"],
        color="green",
        marker="^",
        s=50,
        label="Entrada Largo",
    )
    ax2.scatter(
        shorts.index,
        shorts["z_score"],
        color="red",
        marker="v",
        s=50,
        label="Entrada Corto",
    )

    ax2.set_title("Z-Score y Zonas de Arbitraje")
    ax2.set_ylabel("Desviaciones Estándar (Z)")
    ax2.set_xlabel("Tiempo")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("bollinger_pair_trading.png")


def esperar_cierre_vela(timeframe: int):
    ahora = datetime.now()
    proxima_ejecucion = ahora + timedelta(minutes=timeframe)
    proxima_ejecucion = proxima_ejecucion.replace(second=3, microsecond=0)

    segundos_a_esperar = (proxima_ejecucion - ahora).total_seconds()

    print(
        f"[{ahora.strftime('%H:%M:%S')}] Esperando {segundos_a_esperar:.2f} segundos hasta las {proxima_ejecucion.strftime('%H:%M:%S')}..."
    )
    time.sleep(segundos_a_esperar)


def to_dataframe(candles: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        candles, columns=["datetime", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime").sort_index()

    return df


def fetch_ohlcv_range(
    symbol: str,
    timeframe: str,
    start_ts_ms: int,
    end_ts_ms: int,
    exchange: ccxt.Exchange,
) -> pd.DataFrame:
    all_candles: list[list[float]] = []
    current_since = start_ts_ms
    limit = 100

    while current_since < end_ts_ms:
        candles = exchange.fetch_ohlcv(
            symbol=symbol, timeframe=timeframe, since=current_since, limit=limit
        )
        if not candles:
            break

        all_candles.extend(candles)

        last_ts = candles[-1][0]
        if last_ts >= end_ts_ms:
            break

        current_since = last_ts + 1

        time.sleep(1)

    return to_dataframe(all_candles)


def from_dt_to_ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def send_webhook(url: str, action: str, uuid: str) -> bool:
    payload = {"action": action, "uuid": uuid}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url=url, data=json.dumps(payload), headers=headers, timeout=5
        )

        if response.status_code == 200:
            print(f"Webhook enviado correctamente: {action} {uuid}")
            return True

        else:
            print(f"Error al enviar webhook: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error al enviar webhook: {e}")
        return False


def live_pair_trading_strategy(exchange: ccxt.Exchange):
    # Parámetros Estrategia
    timeframe = 15
    lookback = 166
    entry_z = 2.5
    exit_z = 0.0

    # Configuración Webhooks y Bot
    url_bot = "http://localhost:7503/trade_signal"

    # UUIDs para Activo Y (BTC)
    uuid_Y_long = "f369eeee-12c3-4a05-851c-d97287df79d3"
    uuid_Y_short = "502d1aa4-70ba-4eb4-81a8-90cfd39557ae"

    # UUIDs para Activo X (SOL)
    uuid_X_long = "55a06b44-4934-4d2e-8523-9cf8c625fa81"
    uuid_X_short = "187e9965-0460-4ef6-bf01-d8d5f4e64b1c"

    symbol_Y = "BTC/USDT:USDT"
    symbol_X = "SOL/USDT:USDT"

    # Estado de la posición: 1 (Largo Spread), -1 (Corto Spread), 0 (Plano)
    current_position = 0

    while True:
        esperar_cierre_vela(timeframe=timeframe)

        print(f"\n--- Ejecutando análisis ({datetime.now().strftime('%H:%M:%S')}) ---")

        # Obtener suficientes datos para OLS y Bollinger
        data_from = datetime.now() - timedelta(minutes=timeframe * (lookback + 5))
        data_to = datetime.now().replace(second=0, microsecond=0) - timedelta(seconds=1)

        try:
            df_Y = fetch_ohlcv_range(
                symbol_Y,
                timeframe,
                from_dt_to_ts_ms(data_from),
                from_dt_to_ts_ms(data_to),
                exchange=exchange,
            )
            df_X = fetch_ohlcv_range(
                symbol_X,
                timeframe,
                from_dt_to_ts_ms(data_from),
                from_dt_to_ts_ms(data_to),
                exchange=exchange,
            )
        except Exception as e:
            print(f"Error al obtener datos: {e}")
            continue

        df_combined = pd.concat([df_Y["close"], df_X["close"]], axis=1, join="inner")
        df_combined.columns = ["price_y", "price_x"]

        # Descartar vela actual incompleta
        minuto_actual = datetime.now().replace(second=0, microsecond=0)
        df_combined = df_combined[df_combined.index < pd.Timestamp(minuto_actual)]

        if len(df_combined) < lookback:
            print(f"   ⚠️ Datos insuficientes. Requerimos {lookback}")
            continue

        # 1. Calcular Hedge Ratio (OLS en ventana reciente)
        window_ols = df_combined.iloc[-lookback:]
        model = sm.OLS(
            window_ols["price_y"], sm.add_constant(window_ols["price_x"])
        ).fit()
        hedge_ratio = model.params.iloc[1]

        # 2. Calcular Spread y Z-Score
        df_window = df_combined.iloc[-lookback:]
        spread_series = df_window["price_y"] - (hedge_ratio * df_window["price_x"])

        current_spread = spread_series.iloc[-1]
        moving_avg = spread_series.mean()
        moving_std = spread_series.std()

        if moving_std == 0:
            continue

        zscore = (current_spread - moving_avg) / moving_std

        action_taken = False

        # LÓGICA DE CIERRE DE POSICIONES
        if current_position == 1 and zscore >= -exit_z:
            action_taken = True
            print("   🔴 Cerrando Largo de Spread (Z-Score alcanzó media)")
            send_webhook(url_bot, "closeDeal", uuid_Y_long)
            send_webhook(url_bot, "closeDeal", uuid_X_short)
            current_position = 0

        elif current_position == -1 and zscore <= exit_z:
            action_taken = True
            print("   🔴 Cerrando Corto de Spread (Z-Score alcanzó media)")
            send_webhook(url_bot, "closeDeal", uuid_Y_short)
            send_webhook(url_bot, "closeDeal", uuid_X_long)
            current_position = 0

        # LÓGICA DE APERTURA DE POSICIONES
        if current_position == 0:
            if zscore < -entry_z:
                action_taken = True
                print("   🟢 Abriendo Largo de Spread (Comprar Y, Vender X)")
                send_webhook(url_bot, "startDeal", uuid_Y_long)
                send_webhook(url_bot, "startDeal", uuid_X_short)
                current_position = 1

            elif zscore > entry_z:
                action_taken = True
                print("   🟢 Abriendo Corto de Spread (Vender Y, Comprar X)")
                send_webhook(url_bot, "startDeal", uuid_Y_short)
                send_webhook(url_bot, "startDeal", uuid_X_long)
                current_position = -1

        if not action_taken:
            print("   💤 Sin cambios en las posiciones operativas.")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    MODO = "backtest"

    if MODO == "backtest":
        # Asegúrate de proporcionar la ruta correcta a tus CSV
        try:
            data_btc = pd.read_csv(
                "BTC_USDC:USDC_15m_1769177700000_1771769700000.csv",
                index_col="datetime",
            )
            data_sol = pd.read_csv(
                "SOL_USDC:USDC_15m_1769177700000_1771769700000.csv",
                index_col="datetime",
            )
        except FileNotFoundError:
            print(
                "Error: Archivos CSV no encontrados. Generando datos aleatorios para demostración."
            )
            fechas = pd.date_range("2023-01-01", periods=1000, freq="15min")
            data_btc = pd.DataFrame(
                {"close": np.random.normal(67000, 100, 1000).cumsum()}, index=fechas
            )
            data_sol = pd.DataFrame(
                {"close": np.random.normal(84, 1, 1000).cumsum()}, index=fechas
            )

        exit_z = 0.0

        best_params = run_optimization(data_btc["close"], data_sol["close"], exit_z)

        lookback = best_params["lookback"]
        entry_z = best_params["entry_z"]

        df_signals = bollinger_pair_trading(
            data_btc["close"], data_sol["close"], lookback, entry_z, exit_z
        ).dropna()

        df_backtest = calcular_backtest(
            df_signals, capital_inicial=10000.0, fee_rate=0.001
        )
        metricas = calcular_metricas(df_backtest, periodos_por_ano=35040)

        print("\n--- MÉTRICAS DEL BACKTEST ---")
        for key, value in metricas.items():
            print(f"{key}: {value}")

        print("\n--- MUESTRA DEL DATAFRAME ---")
        print(
            df_backtest[
                [
                    "price_y",
                    "price_x",
                    "net_units",
                    "pos_Y",
                    "pos_X",
                    "pnl_neto",
                    "equity",
                    "hedge_ratio",
                ]
            ].tail(10)
        )

        plot_results(df_backtest, entry_z)

    elif MODO == "live":
        exchange = ccxt.bitget()

        live_pair_trading_strategy(exchange=exchange)
