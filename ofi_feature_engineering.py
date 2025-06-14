import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_best_level_ofi(df):
    ofi = []
    prev_bid_price, prev_bid_size = None, None
    prev_ask_price, prev_ask_size = None, None

    for i in range(len(df)):
        row = df.iloc[i]
        bid_price = row['bid_px_00']
        ask_price = row['ask_px_00']
        bid_size = row['bid_sz_00']
        ask_size = row['ask_sz_00']

        if prev_bid_price is None:
            prev_bid_price, prev_bid_size = bid_price, bid_size
            prev_ask_price, prev_ask_size = ask_price, ask_size
            ofi.append(0)
            continue

        if bid_price > prev_bid_price:
            bid_ofi = bid_size
        elif bid_price == prev_bid_price:
            bid_ofi = bid_size - prev_bid_size
        else:
            bid_ofi = -bid_size

        if ask_price > prev_ask_price:
            ask_ofi = -ask_size
        elif ask_price == prev_ask_price:
            ask_ofi = ask_size - prev_ask_size
        else:
            ask_ofi = ask_size

        ofi.append(bid_ofi - ask_ofi)

        prev_bid_price, prev_bid_size = bid_price, bid_size
        prev_ask_price, prev_ask_size = ask_price, ask_size

    return np.array(ofi)


def compute_multi_level_ofi(df, levels=10):
    ofi = []
    pbp, pap, pbs, pas = [None]*levels, [None]*levels, [None]*levels, [None]*levels

    for i in range(len(df)):
        row = df.iloc[i]
        total_ofi = 0

        for level in range(levels):
            bp, ap = row[f'bid_px_0{level}'], row[f'ask_px_0{level}']
            bs, az = row[f'bid_sz_0{level}'], row[f'ask_sz_0{level}']

            if pbp[level] is None:
                pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az
                continue

            bid_ofi = bs if bp > pbp[level] else (bs - pbs[level] if bp == pbp[level] else -bs)
            ask_ofi = -az if ap > pap[level] else (az - pas[level] if ap == pap[level] else az)
            total_ofi += bid_ofi - ask_ofi

            pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az

        ofi.append(total_ofi)

    return np.array(ofi)


def compute_integrated_ofi(df, levels=10):
    pbp, pap, pbs, pas = [None]*levels, [None]*levels, [None]*levels, [None]*levels
    ofi_matrix = []

    for i in range(len(df)):
        row = df.iloc[i]
        level_ofis = []

        for level in range(levels):
            bp, ap = row[f'bid_px_0{level}'], row[f'ask_px_0{level}']
            bs, az = row[f'bid_sz_0{level}'], row[f'ask_sz_0{level}']

            if pbp[level] is None:
                pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az
                level_ofis.append(0)
                continue

            bid_ofi = bs if bp > pbp[level] else (bs - pbs[level] if bp == pbp[level] else -bs)
            ask_ofi = -az if ap > pap[level] else (az - pas[level] if ap == pap[level] else az)
            level_ofis.append(bid_ofi - ask_ofi)

            pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az

        ofi_matrix.append(level_ofis)

    scaled = StandardScaler().fit_transform(ofi_matrix)
    pca = PCA(n_components=1)
    integrated = pca.fit_transform(scaled)
    return integrated.flatten()

def compute_lagged_cross_asset_ofi(df, ofi_col='best_level_ofi', lag=1):
    return df[ofi_col].shift(lag).fillna(0)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_best_level_ofi(df):
    ofi = []
    prev_bid_price, prev_bid_size = None, None
    prev_ask_price, prev_ask_size = None, None

    for i in range(len(df)):
        row = df.iloc[i]
        bid_price = row['bid_px_00']
        ask_price = row['ask_px_00']
        bid_size = row['bid_sz_00']
        ask_size = row['ask_sz_00']

        if prev_bid_price is None:
            prev_bid_price, prev_bid_size = bid_price, bid_size
            prev_ask_price, prev_ask_size = ask_price, ask_size
            ofi.append(0)
            continue

        if bid_price > prev_bid_price:
            bid_ofi = bid_size
        elif bid_price == prev_bid_price:
            bid_ofi = bid_size - prev_bid_size
        else:
            bid_ofi = -bid_size

        if ask_price > prev_ask_price:
            ask_ofi = -ask_size
        elif ask_price == prev_ask_price:
            ask_ofi = ask_size - prev_ask_size
        else:
            ask_ofi = ask_size

        ofi.append(bid_ofi - ask_ofi)

        prev_bid_price, prev_bid_size = bid_price, bid_size
        prev_ask_price, prev_ask_size = ask_price, ask_size

    return np.array(ofi)


def compute_multi_level_ofi(df, levels=10):
    ofi = []
    pbp, pap, pbs, pas = [None]*levels, [None]*levels, [None]*levels, [None]*levels

    for i in range(len(df)):
        row = df.iloc[i]
        total_ofi = 0

        for level in range(levels):
            bp, ap = row[f'bid_px_0{level}'], row[f'ask_px_0{level}']
            bs, az = row[f'bid_sz_0{level}'], row[f'ask_sz_0{level}']

            if pbp[level] is None:
                pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az
                continue

            bid_ofi = bs if bp > pbp[level] else (bs - pbs[level] if bp == pbp[level] else -bs)
            ask_ofi = -az if ap > pap[level] else (az - pas[level] if ap == pap[level] else az)
            total_ofi += bid_ofi - ask_ofi

            pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az

        ofi.append(total_ofi)

    return np.array(ofi)


def compute_integrated_ofi(df, levels=10):
    pbp, pap, pbs, pas = [None]*levels, [None]*levels, [None]*levels, [None]*levels
    ofi_matrix = []

    for i in range(len(df)):
        row = df.iloc[i]
        level_ofis = []

        for level in range(levels):
            bp, ap = row[f'bid_px_0{level}'], row[f'ask_px_0{level}']
            bs, az = row[f'bid_sz_0{level}'], row[f'ask_sz_0{level}']

            if pbp[level] is None:
                pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az
                level_ofis.append(0)
                continue

            bid_ofi = bs if bp > pbp[level] else (bs - pbs[level] if bp == pbp[level] else -bs)
            ask_ofi = -az if ap > pap[level] else (az - pas[level] if ap == pap[level] else az)
            level_ofis.append(bid_ofi - ask_ofi)

            pbp[level], pap[level], pbs[level], pas[level] = bp, ap, bs, az

        ofi_matrix.append(level_ofis)

    scaled = StandardScaler().fit_transform(ofi_matrix)
    pca = PCA(n_components=1)
    integrated = pca.fit_transform(scaled)
    return integrated.flatten()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("first_25000_rows.csv")

    df['best_level_ofi'] = compute_best_level_ofi(df)
    df['multi_level_ofi'] = compute_multi_level_ofi(df)
    df['integrated_ofi'] = compute_integrated_ofi(df)

    df['cross_asset_ofi_simulated'] = compute_lagged_cross_asset_ofi(df)

    df[['ts_event', 'best_level_ofi', 'multi_level_ofi', 'integrated_ofi','cross_asset_ofi_simulated']].to_csv("ofi_output.csv", index=False)