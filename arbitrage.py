import time
from datetime import datetime, timedelta
import json

import ccxt
import pandas as pd
import requests


bitget = ccxt.bitget()
hyperliquid = ccxt.hyperliquid()


def esperar_cierre_vela(timeframe: int):
    ahora = datetime.now()
    minutos_faltantes = timeframe - (ahora.minute % timeframe)

    proxima_ejecucion = ahora + timedelta(minutes=minutos_faltantes)
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


def live_arbitrage_strategy():
    timeframe = 15
    window = 12
    zscore_umbral = 1.5
    url_bot = "http://localhost:7503/trade_signal"
    uuid_long_hyperliquid = "644a0fd2-7f99-476b-8326-d9dd374c6fdb"
    uuid_short_hyperliquid = "c1b2f3b5-315c-4f2f-846c-0881716818fd"
    uuid_long_bitget = "0da348bf-db0e-4096-b7f3-93b14376def7"
    uuid_short_bitget = "2fb8d52d-93ff-48ba-8886-e32af0723bcd"

    pair_bitget = {
        "symbol": "BTC/USDT:USDT",
        "has_position_long": False,
        "has_position_short": False,
    }
    pair_hyperliquid = {
        "symbol": "BTC/USDC:USDC",
        "has_position_long": False,
        "has_position_short": False,
    }

    while True:
        # Despertar cada minuto
        esperar_cierre_vela(timeframe=timeframe)

        print(f"\n--- Ejecutando análisis ({datetime.now().strftime('%H:%M:%S')}) ---")
        print("   📥 Obteniendo datos de los exchanges...")

        data_from = datetime.now() - timedelta(minutes=timeframe * (window + 3))
        data_to = datetime.now().replace(second=0, microsecond=0) - timedelta(seconds=1)

        try:
            df_bitget = fetch_ohlcv_range(
                symbol=pair_bitget["symbol"],
                timeframe=f"{timeframe}m",
                start_ts_ms=from_dt_to_ts_ms(data_from),
                end_ts_ms=from_dt_to_ts_ms(data_to),
                exchange=bitget,
            )
            df_hyperliquid = fetch_ohlcv_range(
                symbol=pair_hyperliquid["symbol"],
                timeframe=f"{timeframe}m",
                start_ts_ms=from_dt_to_ts_ms(data_from),
                end_ts_ms=from_dt_to_ts_ms(data_to),
                exchange=hyperliquid,
            )

        except Exception as e:
            print(f"Error al obtener datos: {e}")
            continue

        df_combined = pd.concat(
            [df_bitget["close"], df_hyperliquid["close"]], axis=1, join="inner"
        )
        df_combined.columns = ["bitget", "hyperliquid"]

        # Descartamos estrictamente la vela del minuto actual (vela abierta) si CCXT la ha devuelto
        minuto_actual = datetime.now().replace(second=0, microsecond=0)
        df_combined = df_combined[df_combined.index < pd.Timestamp(minuto_actual)]

        if len(df_combined) < window:
            print(
                f"   ⚠️ Datos insuficientes. (Tenemos {len(df_combined)} velas coincidentes, requerimos {window})"
            )
            continue

        df_window = df_combined.iloc[-window:]

        spread_series = df_window["bitget"] - df_window["hyperliquid"]
        current_spread = spread_series.iloc[-1]

        std = spread_series.std()
        if std == 0:
            continue

        zscore = (current_spread - spread_series.mean()) / std

        ultima_vela_ts = df_window.index[-1].strftime("%H:%M:%S")
        print(
            f"   📊 [Vela {ultima_vela_ts}] Spread actual: {current_spread:.2f} | Z-Score: {zscore:.2f}"
        )

        action_taken = False

        if (
            pair_bitget["has_position_long"] or pair_hyperliquid["has_position_short"]
        ) and zscore >= 0:
            action_taken = True

            if pair_bitget["has_position_long"]:
                sent_bit = send_webhook(
                    url=url_bot,
                    action="closeDeal",
                    uuid=uuid_long_bitget,
                )
                if sent_bit:
                    pair_bitget["has_position_long"] = False

            if pair_hyperliquid["has_position_short"]:
                sent_hyp = send_webhook(
                    url=url_bot, action="closeDeal", uuid=uuid_short_hyperliquid
                )
                if sent_hyp:
                    pair_hyperliquid["has_position_short"] = False

        elif (
            pair_bitget["has_position_short"] or pair_hyperliquid["has_position_long"]
        ) and zscore <= 0:
            action_taken = True

            if pair_bitget["has_position_short"]:
                sent_bit = send_webhook(
                    url=url_bot,
                    action="closeDeal",
                    uuid=uuid_short_bitget,
                )
                if sent_bit:
                    pair_bitget["has_position_short"] = False

            if pair_hyperliquid["has_position_long"]:
                sent_hyp = send_webhook(
                    url=url_bot,
                    action="closeDeal",
                    uuid=uuid_long_hyperliquid,
                )
                if sent_hyp:
                    pair_hyperliquid["has_position_long"] = False

        if (
            zscore > zscore_umbral
            and not pair_bitget["has_position_short"]
            and not pair_hyperliquid["has_position_long"]
        ):
            action_taken = True

            sent_bit = send_webhook(
                url=url_bot,
                action="startDeal",
                uuid=uuid_short_bitget,
            )
            if sent_bit:
                pair_bitget["has_position_short"] = True
            sent_hyp = send_webhook(
                url=url_bot, action="startDeal", uuid=uuid_long_hyperliquid
            )
            if sent_hyp:
                pair_hyperliquid["has_position_long"] = True

        elif (
            zscore < -zscore_umbral
            and not pair_bitget["has_position_long"]
            and not pair_hyperliquid["has_position_short"]
        ):
            action_taken = True

            sent_bit = send_webhook(
                url=url_bot, action="startDeal", uuid=uuid_long_bitget
            )
            if sent_bit:
                pair_bitget["has_position_long"] = True
            sent_hyp = send_webhook(
                url=url_bot, action="startDeal", uuid=uuid_short_hyperliquid
            )
            if sent_hyp:
                pair_hyperliquid["has_position_short"] = True

        if not action_taken:
            print("   💤 Sin cambios en las posiciones operativas.")


live_arbitrage_strategy()
