V = {"L1": 0.0, "L2": 0.0}

cnt = 0
while True:
    tmp = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
    delta = abs(tmp - V["L1"])
    V["L1"] = tmp

    tmp = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])
    delta = max(delta, abs(tmp - V["L2"]))
    V["L2"] = tmp

    cnt += 1
    if delta < 1e-4:
        print(V)
        print(cnt)
        break
