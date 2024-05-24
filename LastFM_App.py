import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random, os
import networkx as nx
import pandas as pd
import json
import networkx as nx
import numpy as np
from networkx.algorithms import community
import matplotlib.pyplot as plt
import requests
import matplotlib.patches as mpatches
from collections import Counter
import s3fs
from streamlit_searchbox import st_searchbox
import seaborn as sns
import testfunction_model_inference as tfu
from PIL import Image
from io import BytesIO


AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]
bucket_name = st.secrets["bucket_name"]
url = st.secrets["url"]


country_mapping = {17: 'India', 10: 'China', 0: 'Indonesia', 6: 'Pakistan', 14: 'Bangladesh', 3: 'Japan', 8: 'Philippines', 5: 'Vietnam', 15: 'Iran', 16: 'Turkey', 11: 'Thailand', 7: 'Myanmar', 2: 'South Korea', 13: 'Iraq', 9: 'Afghanistan', 12: 'Saudi Arabia', 1: 'Uzbekistan', 4: 'Yemen'}

flag_urls = {
    "India": "https://flagsapi.com/IN/flat/64.png",
    "China": "https://flagsapi.com/CN/flat/64.png",
    "Indonesia": "https://flagsapi.com/ID/flat/64.png",
    "Pakistan": "https://flagsapi.com/PK/flat/64.png",
    "Bangladesh": "https://flagsapi.com/BD/flat/64.png",
    "Japan": "https://flagsapi.com/JP/flat/64.png",
    "Philippines": "https://flagsapi.com/PH/flat/64.png",
    "Vietnam": "https://flagsapi.com/VN/flat/64.png",
    "Iran": "https://flagsapi.com/IR/flat/64.png",
    "Turkey": "https://flagsapi.com/TR/flat/64.png",
    "Thailand": "https://flagsapi.com/TH/flat/64.png",
    "Myanmar": "https://flagsapi.com/MM/flat/64.png",
    "South Korea": "https://flagsapi.com/KR/flat/64.png",
    "Iraq": "https://flagsapi.com/IQ/flat/64.png",
    "Afghanistan": "https://flagsapi.com/AF/flat/64.png",
    "Saudi Arabia": "https://flagsapi.com/SA/flat/64.png",
    "Uzbekistan": "https://flagsapi.com/UZ/flat/64.png",
    "Yemen": "https://flagsapi.com/YE/flat/64.png"
}

s3 = s3fs.S3FileSystem(anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

flags = {}



@st.cache_data
def load_data():
    with s3.open(f"{bucket_name}/graph_df.csv", "rb") as f:
        graph_df = pd.read_csv(f)

    with s3.open(f"{bucket_name}/target_df.csv", "rb") as f:
        target_df = pd.read_csv(f)

    with s3.open(f"{bucket_name}/node_data.csv", "rb") as f:
        node_data = pd.read_csv(f)

    return graph_df, target_df, node_data


@st.cache_resource
def create_graph(graph_df, target_df, node_data):
    graph = nx.Graph()

    # Add nodes from the graph file
    nodes = set(graph_df['node_1']).union(set(graph_df['node_2']))
    graph.add_nodes_from(nodes)

    # Add edges from the graph file
    edges = graph_df[['node_1', 'node_2']].values
    graph.add_edges_from(edges)

    # Add target information to graph nodes
    target_mapping = dict(target_df[['id', 'target']].values)
    nx.set_node_attributes(graph, target_mapping, 'target')

    # Add music information to graph nodes
    music_mapping = dict(node_data[['id', 'music']].values)
    nx.set_node_attributes(graph, music_mapping, 'music')

    return graph




def plot_neighborhood(graph, node, distance=2):
    neighborhood_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=distance).keys()
    subgraph = graph.subgraph(neighborhood_nodes)

    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(10, 10))

    # Color nodes based on 'target' attribute
    colors = [graph.nodes[node]['target'] for node in subgraph.nodes()]
    country_names = [country_mapping.get(target, 'Unknown') for target in colors]
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color=colors, cmap=plt.cm.tab20, font_size=10,
            font_color="black", font_weight="bold")

    # Create legend
    unique_countries = list(set(country_names))
    legend_labels = [f"{country}" for country in unique_countries]
    legend_patches = [mpatches.Patch(color=plt.cm.tab20(i), label=label) for i, label in enumerate(legend_labels)]
    plt.legend(handles=legend_patches, loc='upper left')

    st.pyplot(plt)


def plot_community(subgraph):
    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(10, 10))

    # Color nodes based on 'target' attribute
    colors = [subgraph.nodes[node]['target'] for node in subgraph.nodes()]
    country_names = [country_mapping.get(target, 'Unknown') for target in colors]
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color=colors, cmap=plt.cm.tab20, font_size=10,
            font_color="black", font_weight="bold")

    # Create legend
    unique_countries = list(set(country_names))
    legend_labels = [f"{country}" for country in unique_countries]
    legend_patches = [mpatches.Patch(color=plt.cm.tab20(i), label=label) for i, label in enumerate(legend_labels)]
    plt.legend(handles=legend_patches, loc='upper left')

    st.pyplot(plt)


def plot_target_distribution(target_df):
    plt.figure(figsize=(8, 6))

    target_df['country'] = target_df['target'].map(country_mapping)

    sns.countplot(data=target_df, x='country')
    plt.title('Distribution of Target Values')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(plt)


def intro_page():
    st.title('LastFM Social Network Classifier')
    st.write('This is an app to showcase network visualization and a machine learning model with 89% accuracy.')
    st.write('I am using a SNAP (Stanford Network Analysis Platform) graph dataset with 27,806 edges and 7,624 nodes. The motivation behind this dashboard and Jupyter notebook is to make the dataset easier to explore for individuals who are new to network analysis and Graph ML.')
    st.text(" ")
    st.text(" ")
    st.image("GephiVisLastFM.png", caption="Gephi Visualization of the dataset, with node size representing the degree of nodes")
    st.text(" ")
    st.text(" ")
    st.markdown("""
    ##### Dataset Description
    This dataset represents a social network of LastFM users collected from the public API in March 2020. It consists of LastFM users from Asian countries, with edges representing mutual follower relationships between them. The vertices have 2 features: a list artists liked by the users and the country of the user. The machine learning model on the 'Inference' tab uses the 'liked artists' attribute and relationships with other users to predict the country location of that user.
    """)
    st.markdown("For more information, you can visit the [dataset page](https://snap.stanford.edu/data/feather-lastfm-social.html).")
    st.text(" ")
    st.text(" ")
    st.markdown("""
    ##### My Changes to the Data
    For intuition, I have mapped the country attribute of nodes, to actual country names (see Country Mapping tab). These country names were somewhat arbitrary as the data source does not provide the name of the country IDs. The most common country IDs in the dataset are the most populus countries. For example, instead of referring to country ('target') 0, I call that country ID 'Indonesia' """)
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")


def eda_page(graph, target_df):
    st.title('Network Exploration')
    st.write('Here, you can explore the dataset visually and interactively.')

    node_list = list(graph.nodes)[1:]

    selected_node = st.selectbox('Display neighbourhood for User ID', node_list)
    show_music = st.checkbox('Show artists IDs that this user has liked')

    if selected_node:
        if selected_node in graph:

            target_attribute = graph.nodes[selected_node].get('target', 'Unknown')
            country = country_mapping.get(target_attribute, 'Unknown')
            st.write(f'Country of node {selected_node}: {country}')

            # Handle cases where country name is not found
            if country != 'Unknown':
                flag_url = flag_urls.get(country)
                if flag_url:  # Check if the flag URL exists
                    response = requests.get(flag_url)
                    if response.status_code == 200:  # Check if the request was successful
                        flags[country] = Image.open(BytesIO(response.content)).resize((40, 30), Image.LANCZOS)
                        country_flag = f'<img src="{flag_url}" alt="{country}" width="5">'
                        st.image(flags[country], width = 40)
                    else:
                        st.write(f"Failed to fetch flag for {country}")
                        country_flag = ''
                else:
                    st.write(f"No flag URL found for {country}")
                    country_flag = ''
            else:
                country_flag = ''


            plot_neighborhood(graph, selected_node)

            # Conditionally display the 'music' attribute of the selected node
            if show_music:
                music_attribute = graph.nodes[selected_node].get('music', 'Unknown')
                st.write(f'Music attribute of node {selected_node}: {music_attribute}')

        else:
            st.write(f'Node {selected_node} not found in the graph.')

    selected_community = st.selectbox('Select a community',
                                      [''] + get_community_list())  # Empty string as the first option

    if selected_community:
        communities_df = load_communities()
        community_nodes = communities_df.loc[communities_df['Community'] == selected_community, 'Nodes'].values[0]
        community_nodes = eval(community_nodes)  # Convert string representation of set to actual set

        # Display KPI-style boxes for community metrics
        subgraph = graph.subgraph(community_nodes)
        assortativity = nx.degree_assortativity_coefficient(subgraph)
        transitivity = nx.transitivity(subgraph)
        community_size = len(subgraph.nodes)

        st.subheader("Community Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Assortativity", value=assortativity)
        col2.metric(label="Transitivity", value=transitivity)
        col3.metric(label="Community Size", value=community_size)

        st.write(f'Visualizing community: {selected_community}')
        plot_community(subgraph)

    st.write("Communities have been identified using the built-in networkx community detection algorithm")

    if st.button('Show Target Distribution'):
        plot_target_distribution(target_df)


def create_new_user_graph(graph, selected_friends, selected_artists):
    # Clone the original graph
    new_graph = graph.copy()

    # Add a new node for the new user
    new_user_id = max(new_graph.nodes) + 1 if new_graph.nodes else 0
    new_graph.add_node(new_user_id)

    # Add edges connecting the new user to selected friends
    for friend_id in selected_friends:
        new_graph.add_edge(new_user_id, friend_id)

    # Add music attribute for the new user
    nx.set_node_attributes(new_graph, {new_user_id: selected_artists}, 'music')

    return new_graph


def inference_page(graph):
    st.title('Country Prediction')
    st.write('Here, you can input data for a new hypothetical user and get predictions from the deployed machine learning model.')

    # Load artist data
    with s3.open(f"{bucket_name}/most_common_artists.csv", "rb") as f:
        artists_df = pd.read_csv(f)

    artist_list = artists_df['Artist'].tolist()
    node_list = list(graph.nodes)[1:]

    selected_artists = st.multiselect('Select artists (by Artist ID', artist_list)
    selected_friends = st.multiselect('Select friends (by User ID)', node_list)

    if st.button('Predict'):
        # Create the new graph with the added user
        new_graph = create_new_user_graph(graph, selected_friends, selected_artists)

        new_node = 999999

        G = graph.copy()

        G.add_node(new_node)
        G.add_edges_from([(new_node, cn) for cn in selected_friends])
        Xtest_newnode = create_X_new_node(G, new_node, selected_artists)

        Xtest_newnode.index = Xtest_newnode.index.astype(str)
        features_dict = Xtest_newnode.iloc[0].to_dict()

        response = requests.post(url, json=features_dict)
        prediction = response.json()

        st.write('Here is the result:')
        predicted_country = country_mapping.get(prediction[0], 'Unknown')
        flag_url = flag_urls.get(predicted_country)
        if flag_url:
            response_flag = requests.get(flag_url)
            flag_image = Image.open(BytesIO(response_flag.content)).resize((40, 30), Image.LANCZOS)
            st.image(flag_image, width = 40)
            st.write(f"Predicted Country: {predicted_country}")
        else:
            st.write(f"Predicted Country: {predicted_country}")

    st.markdown("""#### Model Details:
    I am using a model known as Light Gradient-Boosting Machine which is a tree-based
    machine learning algorithm. 
    
    This model was selected due to its insensitivity to multicolinearity and 
    comparative performance against other alorithms such as logistic regression, 
    support vector machines and others. It was also selected due to computational 
    constraints, which meant that XGboost and GNNs were tested less thoroughly than 
    ideal.
    
    The features I used were degree centrality, the percentage country attribute of 
    neighbours, the most common country of second degree neighbours and the artists 
    that the user liked. Hyperparameters were tuned with grid search, and features 
    were reduced using a customised backwards stepwise process which drops the least 
    important features in each iteration. 
    
    The full model building process can be found in the associated ipython notebook.
    """)

    # Load model report from S3
    with s3.open(f"{bucket_name}/test_classification_report.csv", "rb") as f:
        model_report_df = pd.read_csv(f)

    model_report_df.iloc[:, 0] = model_report_df.iloc[:, 0].replace(country_mapping)
    model_report_df = model_report_df.rename(columns={model_report_df.columns[0]: 'Country'})

    st.markdown("""#### Model Performance Report""")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    pd.set_option('display.max_rows', len(model_report_df))
    st.write(model_report_df)

def create_X_new_node(G, new_node, artists):
    X = []
    # Counting number of artists a user listens to and creating a feature out of it
    num_artists = len(set(artists))

    # Extracting centrality feature
    degree_centrality = nx.degree_centrality(G)[new_node]

    # Turning features into a useful (for dataframe) dictionary format
    features = {'degree_centrality': degree_centrality}

    # Identifying the country percentages for the neighbors
    neighbor_countries = [G.nodes[neighbor]['target'] for neighbor in G.neighbors(new_node) if 'target' in G.nodes[neighbor]]
    if neighbor_countries:
        total_neighbors = len(neighbor_countries)
        country_perc = {country: neighbor_countries.count(country) / total_neighbors for country in set(neighbor_countries)}
        for neigh_country, percentage in country_perc.items():
            feature_name = f"neighbor_{neigh_country}_perc"
            features[feature_name] = percentage

    # Identifying the country distribution among 2nd degree neighbors
    second_degree_neighbors = set()
    for neighbor in G.neighbors(new_node):
        second_degree_neighbors.update(G.neighbors(neighbor))

    second_degree_neighbor_countries = [G.nodes[neighbor]['target'] for neighbor in second_degree_neighbors if 'target' in G.nodes[neighbor]]
    if second_degree_neighbor_countries:
        total_second_degree_neighbors = len(second_degree_neighbor_countries)
        second_degree_country_distribution = {country: second_degree_neighbor_countries.count(country) / total_second_degree_neighbors for country in set(second_degree_neighbor_countries)}
        for neigh_country, percentage in second_degree_country_distribution.items():
            feature_name = f"second_degree_neighbor_{neigh_country}_percentage"
            features[feature_name] = percentage

    # Adding a feature of the most common artists of nodes causes overfitting so I am not doing that here

    for artist in artists:
        features[artist] = 1
    X.append(features)

    # Create DataFrame from features and labels
    X = pd.DataFrame(X)

    return X


def load_communities():
    with s3.open(f"{bucket_name}/allcommunities.csv", "rb") as f:
        communities_df = pd.read_csv(f)
    return communities_df


def get_community_list():
    communities_df = load_communities()
    return communities_df['Community'].tolist()

def country_mapping_page():
    st.title('Country Mapping Table')
    st.write('Here is the table displaying the country mappings:')

    # Create a DataFrame from the country mapping dictionary
    country_mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Country ID', 'Country Name'])
    country_mapping_df = country_mapping_df.sort_values(by='Country ID').reset_index(drop=True)
    st.markdown(country_mapping_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    #st.table(country_mapping_df)

def acknowledgements():
    st.title('Acknowledgements')
    st.markdown("""
    ##### Source (citation)
    **Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models**  
    *Benedek Rozemberczki and Rik Sarkar*  
    In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)*, 2020, pp. 1325–1334.  
    [Link to paper](https://dl.acm.org/doi/10.1145/3340531.3412763)
    """)



def main():
    graph_df, target_df, node_data = load_data()
    graph = create_graph(graph_df, target_df, node_data)

    # App layout
    pages = {
        'Introduction': intro_page,
        'Network Exploration': lambda: eda_page(graph, target_df),
        'Predict': lambda: inference_page(graph),
        'Country Mapping': country_mapping_page,
        'Acknowledgements': acknowledgements
    }

    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))

    # Display selected page
    if selected_page in pages:
        pages[selected_page]()


if __name__ == "__main__":
    main()