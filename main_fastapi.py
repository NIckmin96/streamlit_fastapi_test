import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from fastapi import FastAPI
import requests
import json
import torch

from predict import get_model_rec, get_random_rec, load_model

st.set_page_config(page_title="Movie Recommender", layout="wide")


model = load_model()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#############################################################
###    Define functions to handled the session state data  ###
###    Initialize the required session state values       ###
#############################################################
def add_movie(movie):
    if st.session_state["clicked"]:
        st.session_state["selected_movie_count"] += 1
        # st.session_state["added_movie_ids"].append((movie["item"]))
        st.session_state["added_movie_ids"].append(int(movie["item"]))
        st.session_state["clicked"] = False


def set_status(status):
    st.session_state["status"] = status


# To prevent unexpected events by frequent clicking.
def capture_return(_):
    st.session_state["clicked"] = True


def retry():
    st.session_state["selected_movie_count"] = 0
    st.session_state["added_movie_ids"] = []
    st.session_state["status"] = False


def set_value(key):
    st.session_state[key] = st.session_state["key_" + key]


STATE_KEYS_VALS = [
    ("selected_movie_count", 0),  # main part
    ("added_movie_ids", []),  # main part
    ("status", False),
    ("clicked", False),
    ("input_len", 10),  # sidebar
    ("top_k", 10),  # sidebar
    ("years", (1990, 2010)),  # sidebar
]
for k, v in STATE_KEYS_VALS:
    if k not in st.session_state:
        st.session_state[k] = v


#############################################################
###                  Define Side-bar View                 ###
#############################################################
st.sidebar.title("Setting")
years = st.sidebar.slider(
    "Select a range of years",
    min_value=1900,
    max_value=2015,
    value=st.session_state["years"],
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("years",),
    key="key_years",
)


val = st.sidebar.number_input(
    "How many movies do you want to choose?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["input_len"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("input_len",),
    key="key_input_len",
)

st.sidebar.number_input(
    "How many movies do you want to get recommended?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["top_k"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("top_k",),
    key="key_top_k",
)

st.sidebar.button(
    "START",
    on_click=set_status,
    args=(True,),
    disabled=st.session_state["status"],
)

df = pd.read_csv("poster.csv", sep="\t")
df.fillna("", inplace=True)

#############################################################
###                    Define Main View                   ###
#############################################################

st.title("Movies Recomender with Streamlit")

# When the start button has been clicked from the side-bar
if st.session_state["status"]:
    unique_key = 0
    df = df[~df["item"].isin(st.session_state["added_movie_ids"])]
    df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
    random_movies = df.sample(20)
    no_image = Image.open("placeholder.png")
    selection_container = st.empty()

    # While the number of chosen movies is less than the target number
    if st.session_state["selected_movie_count"] < st.session_state["input_len"]:
        with selection_container.container():
            st.subheader(
                "Please select the favorite movie({}-{}): {}/{}".format(
                    st.session_state["years"][0],
                    st.session_state["years"][1],
                    st.session_state["selected_movie_count"],
                    st.session_state["input_len"],
                )
            )

            for row_index in range(2):
                for col_index, col in enumerate(st.columns(10)):
                    movie = random_movies.iloc[unique_key]
                    title = movie["title"]
                    poster = (
                        movie["poster_link"]
                        if movie["poster_link"]
                        else no_image
                    )
                    unique_key += 1

                    with col:
                        st.image(poster)
                        capture_return(
                            st.checkbox(
                                title,
                                key="mov-{}".format(movie["item"]),
                                on_change=add_movie,
                                args=(movie,),
                            )
                        )
    # When the required number of movies is chosen
    else:
        ## Empty the above view
        selection_container.empty()

        with st.container():
            st.subheader("Random Movies")
            with st.spinner("Wait for it..."):
                data = dict(st.session_state)
                response = requests.post(url='http://localhost:8100/random-movies/', data = json.dumps(data))
                # response로 결과값이 옴
                st.write("FastAPI response result : ", response.text)
                
                random_rec_movies = get_random_rec(response.text)
            ## Render recommended movies
            for col_index, col in enumerate(
                st.columns(st.session_state["top_k"])
            ):
                movie = random_rec_movies.iloc[col_index]
                title = movie["title"]
                poster = (
                    movie["poster_link"] if movie["poster_link"] else no_image
                )

                with col:
                    st.image(poster)
                    st.caption(title)

        with st.container():
            st.subheader("Recommended Movies")
            with st.spinner("Wait for it..."):
                response = requests.post(url="http://localhost:8100/model-movies/", data = json.dumps(dict(st.session_state)))
                model_input = response.text # dictionary를 하나의 text로 출력
                model_input = json.loads(model_input) # json으로 변환
                
                st.subheader(f"FastAPI Response Result!! : {response}")
                st.json(model_input)
                
                s3_rec_movies = get_model_rec(
                    model,
                    model_input["added_movie_ids"],
                    model_input["top_k"],
                )
                # s3_rec_movies = get_model_rec(
                #     model,
                #     st.session_state["added_movie_ids"],
                #     st.session_state["top_k"],
                # )

            ## Render recommended movies
            for col_index, col in enumerate(
                st.columns(st.session_state["top_k"])
            ):
                movie = s3_rec_movies.iloc[col_index]
                title = movie["title"]
                poster = (
                    movie["poster_link"] if movie["poster_link"] else no_image
                )
                with col:
                    st.image(poster)
                    st.caption(title)

        (_, center, _) = st.columns([4, 1, 4])
        with center:
            st.button(
                "Retry",
                on_click=retry,
            )


else:
    st.header("&#9664; Waiting for the setting....")
