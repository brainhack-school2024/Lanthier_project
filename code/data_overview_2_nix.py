# -*- coding: utf-8 -*-
"""
This code is highly based on the codes : data_overview_2.py and example_nix.py from 

This repository contains the datasets described in the publication Brochier et al. (2018). 
Massively parallel recordings in macaque motor cortex during an instructed delayed 
reach-to-grasp task, Scientific Data, 5, 180055. http://doi.org/10.1038/sdata.2018.55

available repository at: https://gin.g-node.org/INT/multielectrode_grasp

These codes were modified and combined by Julie Lanthier to reproduce the figures 6 and 8
of the above mentionned paper using the nix files. Since data_overview_2.py uses the 
Blackrock files, lines for mainpulations of nix files were taking from example_nix.py.

Large parts of both codes were used. These are presented with the mentions below.

#############################################################################
Code for generating the second data figure in the manuscript.

Authors: Julia Sprenger, Lyuba Zehl, Michael Denker


Copyright (c) 2017, Institute of Neuroscience and Medicine (INM-6),
Forschungszentrum Juelich, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#############################################################################
Example code for loading and processing of a recording of the reach-
to-grasp experiments conducted at the Institute de Neurosciences de la Timone
by Thomas Brochier and Alexa Riehle from the NIX files.

Authors: Julia Sprenger, Lyuba Zehl, Michael Denker


Copyright (c) 2017, Institute of Neuroscience and Medicine (INM-6),
Forschungszentrum Juelich, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os

import matplotlib.pyplot as plt
from matplotlib import gridspec, transforms

import numpy as np
import quantities as pq

from neo import SpikeTrain
from neo import utils as neo_utils
from neo.utils import *
from reachgraspio import reachgraspio

import odml.tools

from neo import utils as neo_utils
from neo_utils import load_segment
import odml_utils

# Imports from example_nix.py
from neo import Block, Segment
from neo.io import NixIO
from neo.utils import cut_segment_by_epoch, add_epoch, get_events


# =============================================================================
# Define data and metadata directories and general settings
# =============================================================================


def get_monkey_datafile(monkey):
    if monkey == "Lilou":
        return "l101210-001"  # ns2 (behavior) and ns5 present
    elif monkey == "Nikos2":
        return "i140703-001"  # ns2 and ns6 present
    else:
        return ""


# Enter your dataset directory here
datasetdir = "../multielectrode_grasp/datasets_blackrock/"

chosen_els = {'Lilou': range(3, 97, 7), 'Nikos2': range(1, 97, 7)}
chosen_el = {
    'Lilou': chosen_els['Lilou'][0],
    'Nikos2': chosen_els['Nikos2'][0]}
chosen_unit = 1
trial_indexes = range(14)
trial_index = trial_indexes[0]
chosen_events = ['TS-ON', 'WS-ON', 'CUE-ON', 'CUE-OFF', 'GO-ON', 'SR-ON',
                 'RW-ON', 'WS-OFF']  # , 'RW-OFF'


# =============================================================================
# Load data and metadata for a monkey
# =============================================================================
# CHANGE this parameter to load data of the different monkeys
#monkey = 'Nikos2'
monkey = 'Lilou'

datafile = get_monkey_datafile(monkey)

# =============================================================================
# Load data
#
# As a first step, we load the data file into memory as a Neo object.
# =============================================================================

session_name = datafile

# Specify the path to the recording session to load
session_path = os.path.join("..", "multielectrode_grasp","datasets_nix", session_name + ".nix")

# Open the session for reading
session = NixIO(session_path)

# Read the complete dataset in lazy mode generating all neo objects,
# but not loading data into memory.  The lazy neo structure will contain objects
# to capture all recorded data types (time series at 1000Hz (ns2) and 30kHz (ns6)
# scaled to units of voltage, sorted spike trains, spike waveforms and events)
# of the recording session and return it as a Neo Block. The
# time shift of the ns2 signal (LFP) induced by the online filter is
# automatically corrected for by a heuristic factor stored in the metadata
# (correct_filter_shifts=True).

bl = session.read_block()

# Validate there is only a single Segment present in the block
assert len(bl.segments) == 1

# loading data content of the segment
seg = bl.segments[0]

print("Session loaded.")


# =============================================================================
# Segment the data
#
# =============================================================================
# get start and stop events of trials
start_events = neo_utils.get_events(
    seg,
    **{
    'name': 'TrialEvents',
    'trial_event_labels': 'TS-ON',
    'performance_in_trial': 255})
stop_events = neo_utils.get_events(
    seg,
    **{
    'name': 'TrialEvents',
    'trial_event_labels': 'STOP',
    'performance_in_trial': 255})


# there should only be one event object for these conditions
assert len(start_events) == 1
assert len(stop_events) == 1

# insert epochs between 10ms before TS to 50ms after RW corresponding to trails
ep = neo_utils.add_epoch(
    seg,
    start_events[0],
    stop_events[0],
    pre=-250 * pq.ms,
    post=500 * pq.ms,
    trial_status='complete_trials')
ep.array_annotate(trial_type=start_events[0].array_annotations['belongs_to_trialtype'],
                  trial_performance=start_events[0].array_annotations['performance_in_trial'])


# access single epoch of this data_segment
epochs = neo_utils.get_epochs(seg, **{'trial_status': 'complete_trials'})
assert len(epochs) == 1

# remove spiketrains not belonging to chosen_electrode
seg.spiketrains = seg.filter(targdict={'unit_id': chosen_unit},
                             recursive=True, objects='SpikeTrainProxy')

# use prefiltered data if multiple versions are present
raw_signal = seg.analogsignals[0]
for sig in seg.analogsignals:
    if sig.sampling_rate < raw_signal.sampling_rate:
        raw_signal = sig
seg.analogsignals = [raw_signal]


# replacing the segment with a new segment containing all data
# to speed up cutting of segments
seg = load_segment(seg, load_wavefroms=True)

# only keep the chosen electrode signal in the AnalogSignal object
mask = np.isin(np.asarray(seg.analogsignals[0].array_annotations['channel_ids'], dtype=int),
               chosen_els[monkey])

seg.analogsignals[0] = seg.analogsignals[0][:, mask]

# cut segments according to inserted 'complete_trials' epochs and reset trial times
cut_segments = cut_segment_by_epoch(seg, epochs[0], reset_time=True)

# explicitly adding trial type annotations to cut segments
for i, cut_seg in enumerate(cut_segments):
    cut_seg.annotate(trialtype=epochs[0].array_annotations['belongs_to_trialtype'][i]) 


# =============================================================================
# Define figure and subplot axis for first data overview
# =============================================================================
fig = plt.figure(facecolor='w')
fig.set_size_inches(7.0, 9.9)  # (w, h) in inches
# #(7.0, 9.9) corresponds to A4 portrait ratio

gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    left=0.1,
    bottom=0.05,
    right=0.9,
    top=0.975,
    wspace=0.1,
    hspace=0.1,
    width_ratios=None,
    height_ratios=[2, 1])

ax1 = plt.subplot(gs[0, 0])  # top left
ax2 = plt.subplot(gs[0, 1], sharex=ax1)  # top right
ax3 = plt.subplot(gs[1, 0], sharex=ax1)  # bottom left
ax4 = plt.subplot(gs[1, 1], sharex=ax1)  # bottom right

fontdict_titles = {'fontsize': 9, 'fontweight': 'bold'}
fontdict_axis = {'fontsize': 10, 'fontweight': 'bold'}

# the x coords of the event labels are data, and the y coord are axes
event_label_transform = transforms.blended_transform_factory(ax1.transData,
                                                             ax1.transAxes)

trialtype_colors = {
    'SGHF': 'MediumBlue', 'SGLF': 'Turquoise',
    'PGHF': 'DarkGreen', 'PGLF': 'YellowGreen',
    'LFSG': 'Orange', 'LFPG': 'Yellow',
    'HFSG': 'DarkRed', 'HFPG': 'OrangeRed',
    'SGSG': 'SteelBlue', 'PGPG': 'LimeGreen',
    None: 'black'}

event_colors = {
    'TS-ON': 'indigo', 'TS-OFF': 'indigo',
    'WS-ON': 'purple', 'WS-OFF': 'purple',
    'CUE-ON': 'crimson', 'CUE-OFF': 'crimson',
    'GO-ON': 'orangered', 'GO-OFF': 'orangered',
    'SR-ON': 'darkorange',
    'RW-ON': 'orange', 'RW-OFF': 'orange'}

electrode_cmap = plt.get_cmap('bone')
electrode_colors = [electrode_cmap(x) for x in
                    np.tile(np.array([0.3, 0.7]), int(len(chosen_els[monkey]) / 2))]

time_unit = 'ms'
lfp_unit = 'uV'

# define scaling factors for analogsignals

anasig_std = np.mean(np.std(cut_segments[trial_index].analogsignals[0].rescale(lfp_unit), axis=0))
anasig_offset = 3 * anasig_std


# =============================================================================
# SUPPLEMENTARY PLOTTING functions
# =============================================================================
def add_scalebar(ax, std):
    # the x coords of the scale bar are axis, and the y coord are data
    scalebar_transform = transforms.blended_transform_factory(ax.transAxes,
                                                              ax.transData)
    # adding scalebar
    yscalebar = max(int(std.rescale(lfp_unit)), 1) * getattr(pq, lfp_unit) * 2
    scalebar_offset = -2 * std
    ax.vlines(x=0.4,
              ymin=(scalebar_offset - yscalebar).magnitude,
              ymax=scalebar_offset.magnitude,
              color='k',
              linewidth=4,
              transform=scalebar_transform)
    ax.text(0.4, (scalebar_offset - 0.5 * yscalebar).magnitude,
            ' %i %s' % (yscalebar.magnitude, lfp_unit),
            ha="left", va="center", rotation=0, color='k',
            size=8, transform=scalebar_transform)


# =============================================================================
# PLOT DATA OF SINGLE TRIAL (left plots)
# =============================================================================
# get data of selected trial
selected_trial = cut_segments[trial_index]

# PLOT DATA FOR EACH CHOSEN ELECTRODE
for el_idx, electrode_id in enumerate(chosen_els[monkey]):

    # PLOT ANALOGSIGNALS in upper plot
    chids = np.asarray(cut_segments[0].analogsignals[0].array_annotations['channel_ids'], dtype=int)
    chosen_el_idx = np.where(chids == electrode_id)[0][0]
    anasig = selected_trial.analogsignals[0][:, chosen_el_idx]
    ax1.plot(anasig.times.rescale(time_unit),
             np.asarray(anasig.rescale(lfp_unit))
             + anasig_offset.magnitude * el_idx,
             color=electrode_colors[el_idx])

    # PLOT SPIKETRAINS in lower plot
    spiketrains = selected_trial.filter(
        channel_id=electrode_id, objects=SpikeTrain)
    for spiketrain in spiketrains:
        ax3.plot(spiketrain.times.rescale(time_unit),
                 np.zeros(len(spiketrain.times)) + el_idx, 'k|')

# PLOT EVENTS in both plots
for event_type in chosen_events:
    # get events of each chosen event type
    event_data = get_events(selected_trial,
                                      **{'trial_event_labels': event_type})
    for event in event_data:
        event_color = event_colors[event.array_annotations['trial_event_labels'][0]]
        # adding lines
        for ax in [ax1, ax3]:
            ax.axvline(event.times.rescale(time_unit),
                       color=event_color,
                       zorder=0.5)
        # adding labels
        ax1.text(event.times.rescale(time_unit), 0,
                 event.array_annotations['trial_event_labels'][0],
                 ha="center", va="top", rotation=45, color=event_color,
                 size=8, transform=event_label_transform)

# SUBPLOT ADJUSTMENTS
ax1.set_title('single trial', fontdict=fontdict_titles)
ax1.set_ylabel('electrode id', fontdict=fontdict_axis)
ax1.set_yticks(np.arange(len(chosen_els[monkey])) * anasig_offset)
ax1.set_yticklabels(chosen_els[monkey])

ax1.autoscale(enable=True, axis='y')
plt.setp(ax1.get_xticklabels(), visible=False)  # show no xticklabels
ax3.set_ylabel('electrode id', fontdict=fontdict_axis)
ax3.set_yticks(range(0, len(chosen_els[monkey])))
ax3.set_yticklabels(np.asarray(chosen_els[monkey]))
ax3.set_ylim(-1, len(chosen_els[monkey]))
ax3.set_xlabel('time [%s]' % time_unit, fontdict=fontdict_axis)
# ax3.autoscale(axis='y')


# =============================================================================
# PLOT DATA OF SINGLE ELECTRODE
# =============================================================================
# plot data for each chosen trial
chids = np.asarray(cut_segments[0].analogsignals[0].array_annotations['channel_ids'], dtype=int)
chosen_el_idx = np.where(chids == chosen_el[monkey])[0][0]
for trial_idx, trial_id in enumerate(trial_indexes):
    trial_spikes = cut_segments[trial_id].filter(channel_id=chosen_el[monkey], objects='SpikeTrain')
    trial_type = cut_segments[trial_id].annotations['trialtype']
    trial_color = trialtype_colors[trial_type]

    t_signal = cut_segments[trial_id].analogsignals[0][:, chosen_el_idx]
    # PLOT ANALOGSIGNALS in upper plot
    ax2.plot(t_signal.times.rescale(time_unit),
             np.asarray(t_signal.rescale(lfp_unit))
             + anasig_offset.magnitude * trial_idx,
             color=trial_color, zorder=1)

    for t_data in trial_spikes:
        # PLOT SPIKETRAINS in lower plot
        ax4.plot(t_data.times.rescale(time_unit),
                 np.ones(len(t_data.times)) + trial_idx, 'k|')

    # PLOT EVENTS in both plots
    for event_type in chosen_events:
        # get events of each chosen event type
        event_data = get_events(cut_segments[trial_id], **{'trial_event_labels': event_type})
        for event in event_data:
            color = event_colors[event.array_annotations['trial_event_labels'][0]]
            ax2.vlines(x=event.times.rescale(time_unit),
                       ymin=(trial_idx - 0.5) * anasig_offset,
                       ymax=(trial_idx + 0.5) * anasig_offset,
                       color=color,
                       zorder=2)
            ax4.vlines(x=event.times.rescale(time_unit),
                       ymin=trial_idx + 1 - 0.4,
                       ymax=trial_idx + 1 + 0.4,
                       color=color,
                       zorder=0.5)

# SUBPLOT ADJUSTMENTS
ax2.set_title('single electrode', fontdict=fontdict_titles)
ax2.set_ylabel('trial id', fontdict=fontdict_axis)
ax2.set_yticks(np.asarray(trial_indexes) * anasig_offset)
ax2.set_yticklabels(
    [epochs[0].array_annotations['trial_id'][_] for _ in trial_indexes])
ax2.yaxis.set_label_position("right")
ax2.tick_params(direction='in', length=3, labelleft='off', labelright='on')
ax2.autoscale(enable=True, axis='y')
add_scalebar(ax2, anasig_std)
plt.setp(ax2.get_xticklabels(), visible=False)  # show no xticklabels

ax4.set_ylabel('trial id', fontdict=fontdict_axis)
ax4.set_xlabel('time [%s]' % time_unit, fontdict=fontdict_axis)

start, end = ax4.get_xlim()
ax4.xaxis.set_ticks(np.arange(start, end, 1000))
ax4.xaxis.set_ticks(np.arange(start, end, 500), minor=True)
ax4.set_yticks(range(1, len(trial_indexes) + 1))
ax4.set_yticklabels(np.asarray(
    [epochs[0].array_annotations['trial_id'][_] for _ in trial_indexes]))
ax4.yaxis.set_label_position("right")
ax4.tick_params(direction='in', length=3, labelleft='off', labelright='on')
ax4.autoscale(enable=True, axis='y')

# GENERAL PLOT ADJUSTMENTS
# adjust font sizes of ticks
for ax in [ax4.yaxis, ax4.xaxis, ax3.xaxis, ax3.yaxis]:
    for tick in ax.get_major_ticks():
        tick.label.set_fontsize(10)

# adjust time range on x axis
t_min = np.min([cut_segments[tid].t_start.rescale(time_unit)
                for tid in trial_indexes])
t_max = np.max([cut_segments[tid].t_stop.rescale(time_unit)
                for tid in trial_indexes])
ax1.set_xlim(t_min, t_max)
add_scalebar(ax1, anasig_std)


# =============================================================================
# SAVE FIGURE
# =============================================================================
fname = 'data_overview_2_%s' % monkey
fig_path = 'figures/'
for file_format in ['eps', 'pdf', 'png']:
    fig.savefig(fig_path + fname + '.%s' % file_format, dpi=400, format=file_format)
