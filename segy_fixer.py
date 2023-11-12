import streamlit as st
import pandas as pd
import numpy as np
import segyio
import matplotlib.pyplot as plt
import re

temp_in = "temp-in"
temp_out = "temp-out"
if "sgy" not in st.session_state:
    st.session_state["sgy"] = None
if "data" not in st.session_state:
    st.session_state["data"] = None


def parse_trace_headers(segyfile, n_traces):
    """
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    """
    # Get all header keys
    headers = segyio.tracefield.keys
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1), columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
    return df


def parse_text_header(segyfile):
    """
    Format segy text header into a readable, clean dict
    """
    raw_header = segyio.tools.wrap(segyfile.text[0])
    # Cut on C*int pattern
    cut_header = re.split(r"C ", raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace("\n", " ") for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, "0")
        i += 1
        clean_header[key] = item
    return clean_header


def plot_segy(file):
    # Load data
    with segyio.open(file, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]
    # Plot
    plt.style.use("ggplot")  # Use ggplot styles for all plotting
    vm = np.percentile(data, 98)
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    extent = [1, n_traces, twt[-1], twt[0]]  # define extent
    ax.imshow(data.T, cmap="RdBu", vmin=-vm, vmax=vm, aspect="auto", extent=extent)
    ax.set_xlabel("CDP number")
    ax.set_ylabel("TWT [ms]")
    ax.set_title(f"{file}")
    return fig


def write_data_to_session_state(data):
    with open(
        temp_in, "wb"
    ) as out:  ## Open temporary file as bytes and store to temp location
        out.write(data.read())
    with segyio.open(temp_in, ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        th = parse_text_header(f)
        bh = parse_trace_headers(f, n_traces)
        st.session_state["data"] = f.trace.raw[:]
    return n_traces, sample_rate, n_samples, twt, th, bh


def copy_data_to_new_sgy(indices):
    with segyio.open(temp_in, ignore_geometry=True) as src:
        spec = segyio.tools.metadata(src)
        spec.tracecount = len(indices)
        with segyio.create(temp_out, spec) as dst:
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]
            dst.bin = src.bin
            dst.header = src.header
            dst.trace = st.session_state["data"][indices]
    st.write(
        f"Keeping {len(indices)} of {len(st.session_state['data'])} unique traces from  \n {st.session_state['sgy'].name}  \n  to   \n {clean_segy_name(st.session_state['sgy'].name)}"
    )


def display_diagnostics(traces, samples, sample_rate, twt, text_head: dict):
    st.sidebar.write(
        f"Diagnostics:  \nN Traces: {traces}, N Samples: {samples}, Sample rate: {sample_rate}ms, Trace length: {max(twt)}"
    )
    checkbox = st.sidebar.checkbox("Display text header")
    if checkbox:
        text_header_pretty = ""
        for k, i in text_head.items():
            text_header_pretty += f"{k}: {i}  \n"
        st.sidebar.write(f"{text_header_pretty}")


def clean_segy_name(name: str):
    n, *_, ext = name.split(".")
    return n + "_clean." + ext


st.title("SEGY Duplicate Trace Cleaner")
st.session_state["sgy"] = st.file_uploader(
    "First upload SEGY file (might not work with large files)", type=["sgy", "segy"]
)

if st.session_state["sgy"] is not None:
    (
        n_traces,
        sample_rate,
        n_samples,
        twt,
        text_head,
        trace_head,
    ) = write_data_to_session_state(st.session_state["sgy"])
    display_diagnostics(n_traces, n_samples, sample_rate, twt, text_head)
    if st.button("Delete duplicate traces and generate clean segy"):
        u, indices = np.unique(st.session_state["data"], return_index=True, axis=0)
        n_duplicates = len(st.session_state["data"]) - len(u)
        if n_duplicates == 0:
            st.write("No duplicate traces found, returned segy is identical to input")
        else:
            st.write(f"{n_duplicates} duplicates found")
        #   st.dataframe(trace_head)
        df = trace_head[
            not trace_head.duplicated(subset=["CDP_X", "CDP_Y"], keep="first")
        ]
        if len(df) == 0:
            st.write("No duplicate CDP X-Y combination found")
        else:
            st.write("Duplicate CDP X-Y headers:")
            st.write(f"{len(indices)} vs {len(df.index.to_numpy())}")
            st.dataframe(df)
        copy_data_to_new_sgy(indices)

        with open(temp_out, "rb") as out:
            st.download_button(
                label="Download clean segy file",
                data=out,
                file_name=f"{clean_segy_name(st.session_state['sgy'].name)}",
                mime="application/octet-stream",
                key="download-sgy",
            )
    st.divider()
    if st.button("Plot input segy"):
        fig = plot_segy(temp_in)
        st.pyplot(fig)
