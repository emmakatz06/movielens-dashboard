import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MovieLens Ratings Dashboard", layout="wide")

UNKNOWN_GENRE = "unknown"


@st.cache_data
def load_ratings(path: str = "movie_ratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def split_genres(frame: pd.DataFrame) -> pd.DataFrame:
    exploded = frame.copy()
    exploded["genre"] = exploded["genres"].fillna("").str.split("|")
    exploded = exploded.explode("genre")
    exploded["genre"] = exploded["genre"].str.strip()
    exploded = exploded[exploded["genre"].ne("") & exploded["genre"].str.lower().ne(UNKNOWN_GENRE)]
    return exploded


def movie_rating_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("movie_id", as_index=False)
        .agg(
            title=("title", "first"),
            mean_rating=("rating", "mean"),
            n_ratings=("rating", "size"),
        )
    )


def top_movies(stats: pd.DataFrame, min_ratings: int, n: int = 5) -> pd.DataFrame:
    ranked = stats.loc[stats["n_ratings"] >= min_ratings].sort_values(
        ["mean_rating", "n_ratings", "title"],
        ascending=[False, False, True],
    )
    out = ranked.head(n).copy()
    out["mean_rating"] = out["mean_rating"].round(3)
    return out


def horizontal_bar(data: pd.DataFrame, x: str, y: str, title: str, x_label: str, y_label: str):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        title=title,
        labels={x: x_label, y: y_label},
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": data[y].tolist()},
        margin=dict(l=20, r=20, t=50, b=40),
        height=520,
    )
    return fig


ratings = load_ratings()
movies = ratings.drop_duplicates(subset="movie_id")
movie_stats = movie_rating_stats(ratings)
movies_by_genre = split_genres(movies[["movie_id", "genres"]])
ratings_by_genre = split_genres(ratings[["movie_id", "rating", "genres"]])

genre_counts = (
    movies_by_genre.groupby("genre", as_index=False)
    .agg(n_movies=("movie_id", "nunique"))
    .sort_values("n_movies", ascending=True)
)

genre_means = (
    ratings_by_genre.groupby("genre", as_index=False)
    .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
    .sort_values("mean_rating", ascending=True)
)

ratings_with_year = ratings.dropna(subset=["year"]).copy()
ratings_with_year["year"] = ratings_with_year["year"].astype(int)
year_min = int(ratings_with_year["year"].min())
year_max = int(ratings_with_year["year"].max())
max_movie_n = int(movie_stats["n_ratings"].max())

st.title("MovieLens Ratings Dashboard")
st.caption(
    "Each row in the source file is a user rating. Movies are identified by "
    "`movie_id`. Genre analyses omit the `unknown` tag."
)

st.header("1. Genre distribution among rated movies")
st.markdown(
    "Movies can belong to more than one genre (pipe-separated in the data). "
    "Each movie is counted once per listed genre, so genre totals overlap and "
    "percentages would add up to more than 100%."
)

fig_genre_counts = horizontal_bar(
    genre_counts,
    x="n_movies",
    y="genre",
    title="Number of distinct rated movies per genre",
    x_label="Number of movies",
    y_label="Genre",
)
st.plotly_chart(fig_genre_counts, width="stretch")

st.header("2. Average rating by genre")
st.markdown(
    "Average rating is computed from every rating of movies that include that "
    "genre. A single rating of a multi-genre movie contributes to each of its genres."
)

fig_genre_means = horizontal_bar(
    genre_means,
    x="mean_rating",
    y="genre",
    title="Average rating by genre",
    x_label="Average rating (1–5)",
    y_label="Genre",
)
fig_genre_means.update_xaxes(range=[genre_means["mean_rating"].min() - 0.1, 5])
st.plotly_chart(fig_genre_means, width="stretch")

st.header("3. Mean rating by movie release year")
st.markdown(
    "This chart uses each movie's release year (`year`), not the year the "
    "rating was submitted. Movies with a missing release year are excluded "
    "from this chart only."
)

year_start, year_end = st.slider(
    "Release year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

year_filtered = ratings_with_year.loc[
    ratings_with_year["year"].between(year_start, year_end)
]
mean_by_year = (
    year_filtered.groupby("year", as_index=False)
    .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
    .sort_values("year")
)

fig_year = px.line(
    mean_by_year,
    x="year",
    y="mean_rating",
    markers=True,
    title="Mean rating by movie release year",
    labels={"year": "Release year", "mean_rating": "Mean rating (1–5)"},
)
fig_year.update_layout(margin=dict(l=20, r=20, t=50, b=40), height=420)
fig_year.update_yaxes(range=[1, 5])
st.plotly_chart(fig_year, width="stretch")

if mean_by_year.empty:
    st.info("No ratings fall in the selected year range.")

st.header("4. Top 5 best-rated movies")
st.markdown(
    "Movies are ranked by average rating, using `movie_id` as the identifier. "
    "A minimum rating count reduces the effect of movies with only a few reviews. "
    "Ties are broken by rating count, then title."
)

min_ratings = st.slider(
    "Minimum number of ratings",
    min_value=1,
    max_value=max_movie_n,
    value=50,
)

slider_top = top_movies(movie_stats, min_ratings)
slider_plot = slider_top.sort_values("mean_rating", ascending=True)

fig_top = px.bar(
    slider_plot,
    x="mean_rating",
    y="title",
    orientation="h",
    text="n_ratings",
    title=f"Top 5 movies with at least {min_ratings} ratings",
    labels={"mean_rating": "Average rating (1–5)", "title": "Movie", "n_ratings": "Rating count"},
)
fig_top.update_traces(texttemplate="n=%{text}", textposition="outside")
fig_top.update_layout(
    yaxis={"categoryorder": "array", "categoryarray": slider_plot["title"].tolist()},
    margin=dict(l=20, r=80, t=50, b=40),
    height=360,
    xaxis_range=[1, 5.4],
)
st.plotly_chart(fig_top, width="stretch")
st.dataframe(
    slider_top.rename(
        columns={
            "movie_id": "Movie ID",
            "title": "Title",
            "mean_rating": "Average rating",
            "n_ratings": "Number of ratings",
        }
    ),
    hide_index=True,
    width="stretch",
)

if len(slider_top) < 5:
    st.caption(f"Fewer than 5 movies have at least {min_ratings} ratings.")

st.subheader("Required comparison: 50 vs 150 ratings")
col50, col150 = st.columns(2)

with col50:
    st.markdown("**At least 50 ratings**")
    top50 = top_movies(movie_stats, 50)
    st.dataframe(
        top50.rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Title",
                "mean_rating": "Average rating",
                "n_ratings": "Number of ratings",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with col150:
    st.markdown("**At least 150 ratings**")
    top150 = top_movies(movie_stats, 150)
    st.dataframe(
        top150.rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Title",
                "mean_rating": "Average rating",
                "n_ratings": "Number of ratings",
            }
        ),
        hide_index=True,
        width="stretch",
    )
