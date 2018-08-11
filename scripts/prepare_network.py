# coding: utf-8

import logging
logger = logging.getLogger(__name__)
import pandas as pd
idx = pd.IndexSlice

import numpy as np
import scipy as sp
import xarray as xr
import re

from six import iteritems
import geopandas as gpd

import pypsa

def add_co2limit(n, Nyears=1.):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config['electricity']['co2limit'] * Nyears)

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    assert False, "Needs to be fixed, adds NAN"

    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def set_line_s_max_pu(n):
    # set n-1 security margin to 0.5 for 45 clusters and to 0.7 from 200 clusters
    n_clusters = len(n.buses)
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 45) / (200 - 45), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu

    dc_b = n.links.carrier == 'DC'
    n.links.loc[dc_b, 'p_max_pu'] = s_max_pu
    n.links.loc[dc_b, 'p_min_pu'] = - s_max_pu

def set_line_volume_limit(n, lv):
    # Either line_volume cap or cost
    n.lines['capital_cost'] = 0.
    n.links['capital_cost'] = 0.

    if lv > 1.0:
        lines_s_nom = n.lines.s_nom.where(
            n.lines.type == '',
            np.sqrt(3) * n.lines.num_parallel *
            n.lines.type.map(n.line_types.i_nom) *
            n.lines.bus0.map(n.buses.v_nom)
        )

        n.lines['s_nom_min'] = lines_s_nom
        n.links['p_nom_min'] = n.links['p_nom']

        n.lines['s_nom_extendable'] = True
        n.links['p_nom_extendable'] = True

        n.line_volume_limit = lv * ((lines_s_nom * n.lines['length']).sum() +
                                    n.links.loc[n.links.carrier=='DC'].eval('p_nom * length').sum())

    return n

def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='37', lv='2', opts='Co2L-3H'),
            input=['networks/{network}_s{simpl}_{clusters}.nc'],
            output=['networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    opts = snakemake.wildcards.opts.split('-')

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.sum()/8760.

    set_line_s_max_pu(n)

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    else:
        logger.info("No resampling")

    if 'Co2L' in opts:
        add_co2limit(n, Nyears)
        # add_emission_prices(n, exclude_co2=True)

    # if 'Ep' in opts:
    #     add_emission_prices(n)

    # set_line_volume_limit(n, float(snakemake.wildcards.lv))

    solve_opts = snakemake.config['solving']['options']

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               marginal_cost=1.0e5,
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e6)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 + 2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    n.generators['class'] = n.generators['carrier']
    for carrier in n.generators.carrier.unique():
        ind = n.generators.index[n.generators['carrier'] == carrier]
        ind_t = n.generators_t.p_max_pu.columns.intersection(ind)
        if len(ind_t) != 0 and len(ind) != len(ind_t):
            n.generators.loc[ind_t, 'class'] = n.generators.loc[ind_t, 'carrier'] + '_t'

    n.storage_units['class'] = n.storage_units['carrier']
    for carrier in n.storage_units.carrier.unique():
        ind = n.storage_units.index[n.storage_units['carrier'] == carrier]
        ind_t = n.storage_units_t.inflow.columns.intersection(ind)
        if len(ind_t) != 0 and len(ind) != len(ind_t):
            n.storage_units.loc[ind_t, 'class'] = n.storage_units.loc[ind_t, 'carrier'] + '_t'

    n.export_to_netcdf(snakemake.output[0])
