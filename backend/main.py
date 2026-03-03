def run_digital_twin():
    try:
        r = requests.get(THINGSPEAK_URL, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "feeds" not in data or len(data["feeds"]) == 0:
            return {"error": "No feeds available from ThingSpeak"}

        feed = data["feeds"][-1]

    except Exception as e:
        return {"error": f"ThingSpeak error: {str(e)}"}

    timestamp = datetime.utcnow().isoformat()
    filename_time = f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"

    raw_output = {
        "timestamp": timestamp,
        "channel_id": CHANNEL_ID,
        "sensors": {}
    }

    processed_output = {
        "timestamp": timestamp,
        "channel_id": CHANNEL_ID,
        "sensors": []
    }

    try:
        for sensor in SENSOR_CONFIG:

            sensor_id = sensor["sensor_id"]

            pressure = safe_float(
                feed.get(sensor["pressure_field"]),
                DEFAULT_PRESSURE
            )

            flow = safe_float(
                feed.get(sensor["flow_field"]),
                DEFAULT_FLOW
            )

            raw_output["sensors"][f"sensor_{sensor_id}"] = {
                "pressure": pressure,
                "flow": flow
            }

            result = predict_leak(pressure, flow)

            leak_lpm = safe_float(result.get("leak_lpm"), 0)
            leak_mm = safe_float(result.get("leak_mm"), 0)
            prob = safe_float(result.get("prob"), 0)

            prescription = {
                "severity": "Normal",
                "action_type": "No action required",
                "priority": 0
            }

            if result.get("leak") == 1:
                size_ratio = leak_mm / 1000
                mag_ratio = leak_lpm / 10800
                pres = get_prescription(size_ratio, mag_ratio)
                prescription = clean_dict(pres)

            meta = SENSOR_METADATA.get(sensor_id, {})

            processed_output["sensors"].append(clean_dict({
                "sensor_numeric_id": sensor_id,
                "pressure_sensor_id": meta.get("pressure_sensor_id"),
                "flow_sensor_id": meta.get("flow_sensor_id"),
                "pipe_id": meta.get("pipe_id"),
                "pressure": round(pressure, 2),
                "flow": round(flow, 2),
                "leak": int(result.get("leak", 0)),
                "probability": round(prob, 4),
                "leak_lpm": leak_lpm,
                "leak_mm": leak_mm,
                "prescription": prescription
            }))

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

    # ✅ SAVE TO AZURE BLOB (OUTSIDE prediction try block)
    try:
        # 1️⃣ Historical raw
        save_to_blob(
            raw_container_client,
            f"{filename_time}_raw.json",
            raw_output
        )

        # 2️⃣ Historical processed
        save_to_blob(
            processed_container_client,
            f"{filename_time}_processed.json",
            processed_output
        )

        # 3️⃣ Overwrite latest.json (LIVE FILE)
        save_to_blob(
            processed_container_client,
            "latest.json",
            processed_output
        )

    except Exception as e:
        return {"error": f"Azure Blob error: {str(e)}"}

    return processed_output
