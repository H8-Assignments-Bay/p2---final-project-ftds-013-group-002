#from copyreg import pickle

# import libraries
import pandas as pd  #
import re  #
from sklearn.metrics.pairwise import euclidean_distances  #
import numpy as np  #
import plotly.express as px  #
import streamlit as st  #
import pickle #
import base64 #
from pathlib import Path #

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sales Dashboard", page_icon=":dart:", layout="wide")

df = pd.read_csv('out.csv')

# Add 'day' column to dataframe
df["day"] = pd.to_datetime(df["date"]).dt.day

# apply link column to the dataframe
df['link'] = '[link]' + '(' + df['product-href'].astype(str) + ')'

# apply product performance column to the dataframe
df['perfm'] = df['product_rating'] * df['rating_amount']

# round product rating
#df['product_rating'] = df['product_rating'].round(1)

# define open model
def open_package(path):
    """
    helper function for loading model
    """
    with open(path, "rb") as package_file:
        package = pickle.load(package_file)
    return package


def preprocessing_hp(product):
    product = product.lower()
    product = re.sub("\W", " ", product)
    return product

vectorizer_hp = open_package("vectorizer_hp.pkl")

product_names = df['product_name']
product_names = pd.DataFrame(product_names)
product_names['product_name'] = product_names['product_name'].astype(str)
product_names['new_product_name']= product_names['product_name'].apply(lambda x: preprocessing_hp(x))
corpus = product_names['new_product_name'].tolist()
product_names_vect = vectorizer_hp.transform(corpus)

def find_similarity_hp(input_vect):
    distance = []
    for f in product_names_vect:
        distance.append(euclidean_distances(input_vect, f).tolist()[0][0])
    distance = pd.Series(distance) #.sort_values()[0:100].index
    distance = distance[distance < np.percentile(distance, 2)].sort_values()
    distance_index = distance.index
    return distance_index


### -------------------------------------------MAINPAGE--------------------------------------------------------
#st.title(":dart: Toko-Hunt")
## Load Company Logo

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' width='400' height='400' class='img-fluid'>".format(
    img_to_bytes("Teal_Dark_Blue_Elegant_Modern_Letter_A_Rocket_Logo-removebg-preview.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)
#st.markdown("##")

##### Text Input And Processing
product_name = st.text_input("Type product name: ")
product_price = st.number_input(label='Product price (Rp.): ',
                min_value=0.0,
                max_value=100000000.0,
                step=1e+5,
                format="%.0g")

product_name = preprocessing_hp(product_name)
product_name_vect = vectorizer_hp.transform([product_name])
idx = find_similarity_hp(product_name_vect)
df_idx = df.iloc[idx]

##### TOP KPI's
price_distribution = df_idx['price']/1000
perfm = df_idx['perfm']
average_price = int(df_idx["price"].mean())
min_price = int(df_idx["price"].min())
max_price = int(df_idx["price"].max())
average_store_rating = round(df_idx["product_rating"].mean(), 1)
star_rating = ":star:" * int(round(average_store_rating, 0))
product_sold = round(df_idx["product_sold"].sum(), 2)
best_store = df_idx[df_idx['perfm'] == df_idx['perfm'].max()]['store_name'].tolist()[0]

left_column, mid_column, right_column = st.columns(3)
left_column.metric(f"Average Price (Range: Rp.{int(min_price/1000)}K-{int(max_price/1000)}K)", f"Rp.{average_price}")
mid_column.metric("Your Price", f"Rp.{int(product_price)}", f"Save Rp.{average_price-product_price}")
right_column.metric("Popular store", f"{best_store}")

#with left_column:
 #   st.write(f"Avg: Rp.{average_price}")
   # st.subheader("Average price:")
   # st.subheader(f"Rp.{average_price:,}")
#with mid_column:
#    st.subheader("Products Sold:")
#    st.subheader(f"{product_sold}")
#with right_column:
#   st.subheader("Average Store Rating:")
#    st.subheader(f"{average_store_rating} {star_rating}")

## Show Dataframe
st.subheader("Below are similar products to hunt!")
st.write(f"Keywords: {product_name} -- Price: Rp.{product_price}")
st.dataframe(df_idx[['product_name', 'price', 'product_rating', 'rating_amount','product_sold', 'store_location', 'store_name']].style.format({'product_rating':'{:.1f}', 'store_rating':'{:.1f}'}), width=2000, height=400)


### -----------------------------------------------SIDEBAR----------------------------------------------------
st.sidebar.header("Please Filter Here:")

product_indexes = st.sidebar.multiselect(
    "Select Index Bracket:",
    options=df_idx.index,
    default= None,
)

for product_index in product_indexes:
    st.sidebar.write(f"Checkout the Store: [{product_index}]({df_idx['product-href'][product_index]})")

st.markdown("""---""")

#df_selection = df.query(
    #"store_location == @store_loc & category == @product_categories" # & Customer_type ==@customer_type & Gender == @gender"
#)

#st.dataframe(df_selection)

### --------------------------------------------MORE INFO------------------------------------------------

st.subheader("Price Distribution (in K rupiahs)")
st.bar_chart(price_distribution, height=250, use_container_width=True)

st.subheader("Product Performance (rating * reviews)")
st.bar_chart(perfm, height=250, use_container_width=True)

sales_by_date = df_idx.groupby(by=["day"]).sum()[["product_sold"]]
fig_daily_sales = px.bar(
    sales_by_date,
    x=sales_by_date.index,
    y="product_sold",
    title="<b>Sales by date</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_date),
    template="plotly_white",
)
fig_daily_sales.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)
st.plotly_chart(fig_daily_sales, use_container_width=True)

st.markdown("""---""")


# ---- HIDE STREAMLIT STYLE ----
#hide_st_style = """
#            <style>
            #MainMenu {visibility: hidden;}
#            footer {visibility: hidden;}
#            header {visibility: hidden;}
#            </style>
#           """
#st.markdown(hide_st_style, unsafe_allow_html=True)