"""Tournament bot — plugs trained MLP + agent recovery into the room WebSocket.

Usage:
    python my_bot.py --room <room-name> --name <your-name>
    python my_bot.py --room final2026 --name Keidi
"""
import argparse
import json
import threading
import time
import numpy as np
import websocket

from drive2win.agent import make_policy

WEIGHTS  = "nav_filtered_mirrored.npz"   # swap to your best model
HOST     = "ml.ferit.tech"


def obs_to_state(obs: dict, speed: float = 0.0) -> dict:
    """Convert tournament room obs → the state format make_policy expects.

    The tournament server only sends position/navigation — no speed or rays.
    Speed is computed externally from position deltas and passed in.
    Rays default to 50.0 (clear) since the server does not provide them.
    """
    nav = obs.get("navigation", {})
    sensors = {
        "speed":               speed,
        "heading_error":       nav.get("heading_error", 0.0),
        "checkpoint_distance": nav.get("distance",    100.0),
        "rays":                [50.0] * 8,
        "ground_friction":     1.0,
    }
    state = {"sensors": sensors}
    if "position" in obs:
        state["position"] = obs["position"]
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room",    default="keidi")
    ap.add_argument("--name",   default="Keidi")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--host",   default=HOST)
    ap.add_argument("--hz",     type=float, default=20.0)
    args = ap.parse_args()

    policy = make_policy(args.weights)
    print(f"Loaded weights: {args.weights}")

    url = f"wss://{args.host}/ws/room/bot?room={args.room}&name={args.name}"
    print(f"Connecting to {url}")

    latest_obs   = {"obs": None}
    obs_lock     = threading.Lock()
    running      = [True]
    race_active  = [False]
    bot_key      = [None]

    def on_open(ws):
        print(f"[{args.name}] connected — sending ready")
        ws.send(json.dumps({"type": "ready", "ready": True}))

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")

        if t == "bot_assigned":
            bot_key[0] = msg.get("bot_key")
            print(f"[{args.name}] bot_key={bot_key[0]}")

        elif t == "round_start":
            print(f"[{args.name}] round_start idx={msg.get('round_index')} "
                  f"seed={msg.get('seed')} obstacles={msg.get('obstacles')}")
            race_active[0] = True

        elif t == "round_end":
            print(f"[{args.name}] round_end idx={msg.get('round_index')}")
            race_active[0] = False

        elif t == "tournament_end":
            print(f"[{args.name}] tournament finished!")
            standings = msg.get("standings", [])
            for r in standings:
                print(f"  #{r.get('rank')} {r.get('name')} cps={r.get('total_checkpoints')}")
            running[0] = False

        elif t in ("state", "state_update"):
            bots = msg.get("bots", [])
            if isinstance(bots, list):
                my_bot = next((b for b in bots if b.get("bot_key") == bot_key[0]), None)
                if my_bot is None and bots:
                    my_bot = bots[0]
            else:
                my_bot = bots.get(bot_key[0]) or (next(iter(bots.values())) if bots else None)
            if my_bot:
                if latest_obs["obs"] is None:
                    print(f"  [debug] bot keys: {list(my_bot.keys())}")
                    print(f"  [debug] bot sample: {my_bot}")
                with obs_lock:
                    latest_obs["obs"] = my_bot
                    race_active[0] = True

        elif t == "error":
            print(f"[{args.name}] error: {msg.get('code')} {msg.get('message')}")
        else:
            print(f"[{args.name}] unhandled msg: {t} keys={list(msg.keys())}")

    def on_error(ws, err):
        print(f"[{args.name}] ws error: {err}")

    def on_close(ws, code, reason):
        running[0] = False
        print(f"[{args.name}] disconnected ({code} {reason})")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    time.sleep(0.8)

    period = 1.0 / args.hz
    prev_pos = [None]
    print(f"Waiting for race to start...")

    try:
        while running[0]:
            with obs_lock:
                obs = latest_obs["obs"]

            if obs and race_active[0]:
                pos = obs.get("position", {})
                if prev_pos[0] is not None and pos:
                    dx = pos.get("x", 0) - prev_pos[0].get("x", 0)
                    dz = pos.get("z", 0) - prev_pos[0].get("z", 0)
                    speed = float(np.sqrt(dx * dx + dz * dz) * args.hz)
                else:
                    speed = 0.0
                prev_pos[0] = pos
                throttle, steering = policy(obs_to_state(obs, speed))
            else:
                throttle, steering = 0.7, 0.0

            try:
                ws.send(json.dumps({
                    "type":     "control",
                    "throttle": float(np.clip(throttle, -1, 1)),
                    "steering": float(np.clip(steering, -1, 1)),
                }))
            except Exception:
                break

            time.sleep(period)

    except KeyboardInterrupt:
        print(f"\n[{args.name}] stopped")
    finally:
        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
