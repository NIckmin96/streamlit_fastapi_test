import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_poster(title: str):
    image_link = ""
    try:

        domain = "http://www.imdb.com"
        # year = title[title.rfind("(") + 1 : -1][:4]
        title = title.lower()
        title = re.sub(r"\([^()]*\)", "", title)
        title = title.replace("  ", " ")
        if title.endswith(", the "):
            title = "the " + title.split(", the ")[0]
        if title.endswith(", la "):
            title = "la " + title.split(", la ")[0]

        title = urllib.parse.quote(title)
        search_url = "{}/find?q={}".format(domain, title)

        res = requests.get(search_url)
        html = res.text
        soup = BeautifulSoup(html, "html.parser")

        headers = soup.find_all("h3", {"class": "findSectionHeader"})
        for index, header in enumerate(headers):

            if header.text == "Titles":
                break

        movie_link = (
            soup.find_all("div", {"class": "findSection"})[index]
            .find("td", {"class": "result_text"})
            .find("a")
            .attrs["href"]
        )
        movie_url = domain + movie_link

        res = requests.get(movie_url)
        html = res.text
        soup = BeautifulSoup(html, "html.parser")
        image_link = soup.find("div", {"class": "ipc-poster"}).find("img", {"class": "ipc-image"}).attrs["src"]
    except Exception as e:
        print("Error")
        print(search_url)

    return image_link


title_df = pd.read_csv("train/titles.tsv", sep="\t")
movies = title_df.values.tolist()

poster_df = pd.DataFrame([], columns=["item", "title", "poster_link", "year"])
poster_df.to_csv("poster.csv", index=False, sep="\t")

num_of_workers = 10

for index in range(0, len(movies), num_of_workers):
    list_of_movies = movies[index : index + num_of_workers]
    list_of_titles = [title for _, title in list_of_movies]
    with ThreadPoolExecutor(max_workers=len(list_of_movies)) as pool:
        response_list = list(pool.map(get_poster, list_of_titles))
    for i in range(len(response_list)):
        list_of_movies[i].append(response_list[i])
    for i in range(len(list_of_titles)):
        left_index = list_of_titles[i].rfind("(")
        right_index = list_of_titles[i].rfind(")")
        list_of_movies[i].append(int(list_of_titles[i][left_index + 1 : right_index].split("-")[0]))

    poster_df = pd.DataFrame(list_of_movies)
    poster_df.to_csv(
        "poster.csv",
        mode="a",
        header=False,
        index=False,
        sep="\t",
    )

# print(get_poster("Thing, The (1982)"))
