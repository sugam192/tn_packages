from integrate import IntegrateWebSocket
import threading

class TickStreamer:
    def __init__(self, conn, exchange, symbol):
        self.conn = conn
        self.exchange = exchange
        self.symbol = symbol
        self.token = self._get_token_for_symbol(exchange, symbol)
        self.latest_tick = {}
        self.latest_ltp = None
        self.iws = IntegrateWebSocket(conn)
        self.on_tick_callback = None  # <-- Add callback attribute
        self._setup_callbacks()
        self._thread = None

    def _get_token_for_symbol(self, exchange, symbol):
        token = next(
            (i["token"] for i in self.conn.symbols if i["segment"] == exchange and i["trading_symbol"] == symbol),
            None,
        )
        if not token:
            print(f"[TickStreamer] ERROR: Token not found for {symbol} in symbols file")
            raise Exception(f"Token not found for {symbol} in symbols file")
        print(f"[TickStreamer] Found token for {symbol}: {token}")
        return (exchange, token)

    def _on_login(self, iws):
        print(f"[TickStreamer] on_login called. Subscribing to {self.token}")
        iws.subscribe(self.conn.SUBSCRIPTION_TYPE_TICK, [self.token])

    def _on_tick_update(self, iws, tick):
        #print(f"[TickStreamer] on_tick_update received: {tick}")
        self.latest_tick = tick
        if 'lp' in tick:
            self.latest_ltp = float(tick['lp'])
            print(f"[TickStreamer] Updated latest_ltp: {self.latest_ltp}")
        # Call user callback if set
        if self.on_tick_callback:
            self.on_tick_callback(self.latest_ltp, tick)

    def _setup_callbacks(self):
        self.iws.on_login = self._on_login
        self.iws.on_tick_update = self._on_tick_update

    def start(self):
        print("[TickStreamer] Starting WebSocket thread...")
        self._thread = threading.Thread(target=lambda: self.iws.connect(ssl_verify=False), daemon=True)
        self._thread.start()

    def get_latest_ltp(self):
        return self.latest_ltp

    def get_latest_tick(self):
        return self.latest_tick