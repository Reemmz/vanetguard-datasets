
"""
VANETGuard v1.0
Disinformation Detection for VANETs using CAM and DENM message coherency and reputation scoring.

Key Concepts:
- Event similarity in time and space
- RSU and CAM message coherence checks
- Reputation update using weighted beta algorithm
- Time complexity commentary included
"""

import argparse
import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from shapely.geometry import shape, Point, MultiPoint
from shapely import centroid
from geopy import distance
from threshold_utils import get_threshold_set_from_type
import time


from iota import Iota, TryteString, ProposedTransaction, Address, Tag

# Connect to IOTA Tangle testnet node (Shimmer or similar)
iota_api = Iota('https://api.testnet.shimmer.network')

def send_to_iota(message_str, tag='VANETGUARD'):
    try:
        tx = ProposedTransaction(
            address=Address(b'TESTIOTAVANETGUARD999999999999999999999999999999999999999999999999999999999999999'),
            message=TryteString.from_unicode(message_str),
            tag=Tag(tag),
            value=0
        )
        result = iota_api.send_transfer(transfers=[tx])
        print("✅ Logged to IOTA Tangle:", result['bundle'].hash)
    except Exception as e:
        print("❌ IOTA Logging failed:", e)

logging.basicConfig(filename='algorithm.log', filemode='a', format='%(levelname)s:%(message)s')
rep_changes = [0] * 6

vanetguard_version = "1.0"

def check_distance(den_pos, cam_pos, r=50):
    return distance.distance(den_pos, cam_pos).m <= r

def check_cov_intersection(geojson, point):
    """
    Checks if a vehicle's point is within the coverage area defined by geojson.
    Time Complexity: O(n), where n = number of polygons in geojson.
    """
    for feature in geojson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return True
    return False

def find_similar_event(event_collector, event, thresholds_type='mean'):
    """
    Checks for a similar event in the collector based on time and space proximity.
    Time Complexity: O(k), where k = number of tracked events.
    """
    eventType, p, t = event
    time_threshold, radius = get_threshold_set_from_type(thresholds_type)[eventType]
    time_threshold *= 1000
    for key in event_collector:
        existing = event_collector[key]
        if existing["eventType"] == eventType and abs(existing["time_centroid"] - t) <= time_threshold:
            if check_distance((p.y, p.x), (existing["space_centroid"].y, existing["space_centroid"].x), r=radius):
                existing["time_centroid"] = (existing["time_centroid"] + t) / 2
                existing["space_centroid"] = centroid(MultiPoint([existing["space_centroid"], p]))
                return key
    return None

def update_reputation(df_reputations, source, reputation_score, alfa, beta):
    """
    Updates reputation score using beta-weighted update rule.
    Time Complexity: O(1) for lookup + update
    """
    old_rep = df_reputations.loc[df_reputations['vehicle_did'] == source]['score'].values[0]
    new_rep = (alfa * old_rep) + (beta * (old_rep + reputation_score))
    new_rep = max(0, min(1, new_rep))  # Clamp between 0 and 1
    df_reputations.loc[df_reputations['vehicle_did'] == source] = [source, new_rep]

def run_vanetguard(args):
    print(f"\n🚗 VANETGuard v{vanetguard_version} starting...\n")
    overall_start = time.perf_counter()
    per_denm_latencies = []

    cam_time_window = args.time_window_cam * 1000
    denm_time_window = args.time_window_denm * 1000
    rsuScore = args.defaultScore * args.weightRsu
    msgCohScore = args.defaultScore * args.weightMsg
    alfa = args.alfa
    beta = args.beta
    output_folder = args.out_folder
    dataset_name = os.path.basename(args.denmdataset)

    thresholds = get_threshold_set_from_type(args.thresholds_type)
    if not thresholds:
        print('❌ Error: Threshold type not found.')
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(args.coverage) as f:
        geojson = json.load(f)

    denm = pd.read_csv(args.denmdataset, sep=';')
    cam = pd.read_csv(args.camdataset, sep=';')
    reputations = pd.read_csv(args.reputation, sep=';').drop(['Unnamed: 0'], axis=1)

    cam['referencePositionLong'] /= 1e7
    cam['referencePositionLat'] /= 1e7

    tai_sync = datetime.strptime('2004-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    utc_tai_sync = datetime.utcfromtimestamp(tai_sync.timestamp())
    start_sim_time = datetime.strptime(args.startTime, '%Y-%m-%d %H:%M:%S')
    new_start_time = (start_sim_time.timestamp() * 1000) - (utc_tai_sync.timestamp() * 1000)

    denm['message_reception_time'] = (denm['simulation_time'] * 1000) + new_start_time
    cam['message_reception_time'] = (cam['simulationTime'] * 1000) + new_start_time

    denm = denm[denm['situation_informationQ'] > 0.6].drop_duplicates()
    cam = cam.drop_duplicates()

    denm = denm.dropna(subset=['eventPos_long','eventPos_lat'])
    cam = cam.dropna(subset=['referencePositionLat','referencePositionLong'])

    denm = denm.sort_values(by='message_reception_time')
    cam = cam.sort_values(by='message_reception_time')

    event_collector = {}

    for i, row in denm.iterrows():
        start_denm = time.perf_counter()

        reputation_score = 0
        message_age = row['message_reception_time'] - row['detection_time']
        time_threshold = thresholds[row['situation_eventType']][0] * 1000

        if message_age > time_threshold:
            continue

        if (new_start_time + denm_time_window) < row['message_reception_time'] or i == (len(denm) - 1):
            new_start_time = row['message_reception_time']

        point = Point(row['eventPos_long'], row['eventPos_lat'])
        if not check_cov_intersection(geojson, point):
            reputation_score -= msgCohScore
            rep_changes[2] += 1
        else:
            reputation_score += msgCohScore / 2

        cam_subset = cam[(cam['source'] == row['source']) &
                         (cam['message_reception_time'] < row['detection_time']) &
                         (cam['message_reception_time'] > row['detection_time'] - cam_time_window)]

        coherent = 0
        for _, c in cam_subset.iterrows():
            if check_distance((row['eventPos_lat'], row['eventPos_long']),
                              (c['referencePositionLat'], c['referencePositionLong']),
                              r=thresholds[row['situation_eventType']][1]):
                coherent += 1

        if len(cam_subset) > 0:
            percent = (coherent / len(cam_subset)) * 100
            if percent < 10:
                reputation_score -= msgCohScore
                rep_changes[4] += 1
            elif percent >= 30:
                reputation_score += msgCohScore
                rep_changes[5] += 1

        eventType = row['situation_eventType']
        t = row['detection_time']
        existing = find_similar_event(event_collector, (eventType, point, t), thresholds_type=args.thresholds_type)
        if existing:
            if not any(d[3] == row['source'] for d in event_collector[existing]['denms']):
                event_collector[existing]['denms'].append((eventType, point, t, row['source']))
        else:
            key = hash((eventType, point, t))
            event_collector[key] = {
                "eventType": eventType,
                "space_centroid": point,
                "time_centroid": t,
                "denms": [(eventType, point, t, row['source'])]
            }

        
        if reputation_score < 0:
            iota_msg = f"Disinfo alert: vehicle={row['source']}, rep_score={reputation_score:.2f}, event={eventType}, time={t}"
            send_to_iota(iota_msg)

        update_reputation(reputations, row['source'], reputation_score, alfa, beta)

        end_denm = time.perf_counter()
        per_denm_latencies.append((end_denm - start_denm) * 1000)

    out_string = str(beta).split('.')[1]
    reputations.to_csv(f"{output_folder}/new_reputations_{args.thresholds_type}_{dataset_name}_beta_0_{out_string}.csv", sep=';')

    print(f"✅ Finished processing {len(denm)} DENM messages")
    total = time.perf_counter() - overall_start
    print(f"⏱️ Total execution time: {total * 1000:.2f} ms")
    print(f"📊 Per-DENM latency: avg={np.mean(per_denm_latencies):.2f}ms, max={np.max(per_denm_latencies):.2f}ms, min={np.min(per_denm_latencies):.2f}ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VANETGuard - Disinformation Detection in VANETs")
    parser.add_argument('-wc', '--time_window_cam', type=float, default=600)
    parser.add_argument('-wd', '--time_window_denm', type=float, default=20)
    parser.add_argument('-c', '--coverage', type=str, default='coverage.json')
    parser.add_argument('-a', '--alfa', type=float, default=0.5)
    parser.add_argument('-b', '--beta', type=float, default=0.5)
    parser.add_argument('-dd', '--denmdataset', type=str, required=True)
    parser.add_argument('-dc', '--camdataset', type=str, required=True)
    parser.add_argument('-r', '--reputation', type=str, required=True)
    parser.add_argument('-s', '--startTime', type=str, default="2017-06-26 12:00:00")
    parser.add_argument('-l', '--logger', type=str, default='DEBUG')
    parser.add_argument('-tr', '--thresholds_type', type=str, default='mean')
    parser.add_argument('-o', '--out_folder', type=str, default='new_reputations')
    parser.add_argument('-ds', '--defaultScore', type=float, default=0.25)
    parser.add_argument('-wr', '--weightRsu', type=float, default=4)
    parser.add_argument('-wm', '--weightMsg', type=float, default=1)

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.logger.upper()))
    run_vanetguard(args)
