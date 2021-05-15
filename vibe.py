# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:53:39 2021

@author: HP
"""
#import all the required libraries.
# DB
import sqlite3
conn = sqlite3.connect('data2.db')
c = conn.cursor()

# Functions
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS blogtable(author TEXT,article TEXT,postdate DATE)')

def add_data(author,article,postdate):
	c.execute('INSERT INTO blogtable(author,article,postdate) VALUES (?,?,?)',(author,article,postdate))
	conn.commit()

def view_all_notes():
	c.execute('SELECT * FROM blogtable')
	data = c.fetchall()
	return data


import streamlit as st 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from knn_recommender import Recommender
import streamlit.components.v1 as components



footer_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">About Vibe.in</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit,Pandas,Numpy,Matplotlib,Seaborn,Scipy.</p>
	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Us</h5>
	          <ul>
	         
	          <a href="https://www.linkedin.com/in/muskan-thakur-a81752181/" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>
	          </a>
	       
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
          Made by <a class="white-text text-lighten-3" >Muskan Thakur & Shradha Agarwal</a><br/>
	      </div>
	    </div>
	  </footer>
	"""
    

def main():
    
    title_temp ="""
<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
<h4 style="color:white;text-align:center;">{}</h1>
<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
<h6>{}</h6>
<br/>
<br/> 
<p style="text-align:justify">{}</p>
</div>
"""
    html_temp = """
<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:{};text-align:center;">Simple Blog </h1>
</div>
"""
    activities = ["Home","EDA & Recommendations","Add Feedback","View Feedback","About"]
    choice = st.sidebar.selectbox("Menu",activities)
    
               
    if choice == 'EDA & Recommendations':
            st.title("VIBE.IN")
            song_info = pd.read_csv('k.txt',sep='\t',header=None)
            song_info.columns = ['user_id', 'song_id', 'listen_count']
#We are going to use the Million Song Dataset, a freely-available collection of audio features and metadata for a million contemporary popular music tracks.There are two files that will be interesting for us. The first of them will give us information about the songs. Particularly, it contains the user ID, song ID and the listen count. On the other hand, the second file will contain song ID, title of that song, release, artist name and year. We need to merge these two DataFrames. For that aim, we'll use the song_ID
            #Read song  metadata
            song_actual =  pd.read_csv('songss.csv')
            song_actual.drop_duplicates(['song_id'], inplace=True)

           #Merge the two dataframes above to create input dataframe for recommender systems
            songs = pd.merge(song_info, song_actual, on="song_id", how="left")
            songs.head()
            songs.to_csv('songs.csv', index=False)
            df_songs = pd.read_csv('songs.csv')

            st.subheader("Exploring data")
            st.write("First five rows of our dataset")  
            st.dataframe(df_songs.head())
            st.write(f"There are {df_songs.shape[0]} observations in the dataset")
            df_songs.dtypes
            st.write(" Most of the columns contain strings.")
            unique_songs = df_songs['title'].unique().shape[0]
            print(f"There are {unique_songs} unique songs in the dataset")
            st.subheader("Unique songs")
            st.write("There are 2174 unique songs in the dataset")
            unique_artists = df_songs['artist_name'].unique().shape[0]
            print(f"There are {unique_artists} unique artists in the dataset")
            st.subheader("Unique artists")
            st.write("There are 955 unique artists in the dataset")
            unique_users = df_songs['user_id'].unique().shape[0]
            print(f"There are {unique_users} unique users in the dataset")
            st.subheader("Unique users")
            st.write("There are 91 unique users in the dataset")
            st.subheader("Most popular songs")
            ten_pop_songs = df_songs.groupby('title')['listen_count'].count().reset_index().sort_values(['listen_count', 'title'], ascending = [0,1])
            ten_pop_songs['percentage']  = round(ten_pop_songs['listen_count'].div(ten_pop_songs['listen_count'].sum())*100, 2)
            ten_pop_songs = ten_pop_songs[:10]
            st.write("Top ten songs")
            st.dataframe(ten_pop_songs)
            st.write("Visual representation of top 10 songs")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            labels = ten_pop_songs['title'].tolist()
            counts = ten_pop_songs['listen_count'].tolist()
            sns.barplot(x=counts, y=labels, palette='Set3')
            sns.barplot(x=counts, y=labels, palette='Set3')
            st.pyplot()
            st.subheader("Most popular artists")
            ten_pop_artists  = df_songs.groupby(['artist_name'])['listen_count'].count().reset_index().sort_values(['listen_count', 'artist_name'], ascending = [0,1])
            ten_pop_artists = ten_pop_artists[:10]
            st.write("Top ten artists")
            ten_pop_artists
            st.write("Visual representation of top 10 artists")
            labels = ten_pop_artists['artist_name'].tolist()
            counts = ten_pop_artists['listen_count'].tolist()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            a=plt.figure()
            sns.barplot(x=counts, y=labels, palette='Set2')
            sns.despine(left=True, bottom=True)
            st.pyplot(a)
            st.subheader("Listen count by user")
            listen_counts = pd.DataFrame(df_songs.groupby('listen_count').size(), columns=['count'])
            print(f"The maximum time the same user listened to the same songs was: {listen_counts.reset_index(drop=False)['listen_count'].iloc[-1]}")
            st.write("The maximum time the same user listened to the same songs was: 126")
            print(f"On average, a user listen to the same song {df_songs['listen_count'].mean()} times")
            st.write("On average, a user listen to the same song 2.7103333333333333 times")
            st.write("Visual representation of distribution of listen count")
            plt.figure(figsize=(20, 5))
            sns.boxplot(x='listen_count', data=df_songs)
            sns.despine()
            st.pyplot()
            st.write("Visual representation of the most frequent number of times a user listen to the same song")
            listen_counts_temp = listen_counts[listen_counts['count'] > 50].reset_index(drop=False)
            plt.figure(figsize=(16, 8))
            sns.barplot(x='listen_count', y='count', palette='Set3', data=listen_counts_temp)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.show();
            st.pyplot()
            st.write("Visual representation of the songs that a user listens on an average")
            song_user = df_songs.groupby('user_id')['song_id'].count()
            plt.figure(figsize=(16, 8))
            sns.distplot(song_user.values, color='orange')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.show();
            st.pyplot()
            print(f"A user listens to an average of {np.mean(song_user)} songs")
            st.write("A user listens to an average of 32.967032967032964 songs")
            print(f"A user listens to an average of {np.median(song_user)} songs, with minimum {np.min(song_user)} and maximum {np.max(song_user)} songs")
            st.write("A user listens to an average of 19.0 songs, with minimum 1 and maximum 401 songs")
            values_matrix = unique_users * unique_songs
            zero_values_matrix = values_matrix - df_songs.shape[0]
            print(f"The matrix of users x songs has {zero_values_matrix} values that are zero")
            st.write("The matrix of users x songs has 194834 values that are zero")
            st.write("Dealing with such a sparse matrix, take a lot of memory and resources. Therefore, selecting all the users who have listened to at least 15 songs.")
            song_ten_id = song_user[song_user > 15].index.to_list()
            df_song_id_more_ten = df_songs[df_songs['user_id'].isin(song_ten_id)].reset_index(drop=True)
            df_songs_features = df_song_id_more_ten.pivot(index='song_id', columns='user_id', values='listen_count').fillna(0)
            mat_songs_features = csr_matrix(df_songs_features.values)
            st.dataframe(df_songs_features.head())
            df_unique_songs = df_songs.drop_duplicates(subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
            decode_id_song = {
            song: i for i, song in 
            enumerate(list(df_unique_songs.set_index('song_id').loc[df_songs_features.index].title))}
            st.title("RECOMMENDATION")
            model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)
            song = st.text_area("Enter the song you want recommendations for",'I believe in miracles')
            new_recommendations = model.make_recommendation(new_song=song, n_recommendations=10)
            st.subheader(f"The recommendations for {song} ")
            st.subheader(f"{new_recommendations }") 
       
    elif choice == "About":
		          st.title("About App")
		          components.html(footer_temp,height=500)
                  
   
    elif choice == "View Feedback":
		         st.subheader("Feedback")
		         result = view_all_notes()
		
		         for i in result:
			            b_author = i[0]			         
			            b_article = str(i[1])[0:500]
			            b_post_date = i[2]
			            st.markdown(title_temp.format(b_author,b_article,b_post_date),unsafe_allow_html=True)
 
    elif choice == "Add Feedback":
		    st.subheader("Your experience")
		    create_table()
		    blog_author = st.text_input("Enter your Name",max_chars=50)
		    blog_article = st.text_area("Post your comments",height=200)
		    blog_post_date = st.date_input("Date")
		    if st.button("Add"):
			          add_data(blog_author,blog_article,blog_post_date)
			          st.success("Post saved")	

    
    

    else:
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">VIBE.IN</h1>
        <h2 style="color:white;text-align:center;">Music Recommeder</h2>
        
		</div>
		"""
        components.html(html_temp)
        components.html("""
			 <img src="https://www.w3schools.com/w3css/img_md_music.jpg" style="center-align: middle;float:left;width: 100px;height: 100px" >
             <img src="https://i.pinimg.com/originals/47/e0/5b/47e05bb4ebebdd0bed92d5402abb5849.png" style="center-align: middle;float:left;width: 100px;height: 100px" >
              <img src="https://miro.medium.com/max/3200/1*aXpZ--zKN36YWET77Pdj3Q.png" style="center-align: middle;float:left;width: 120px;height: 100px" >
              <img src="https://www.videvo.net/wp-content/uploads/2018/05/Muisc-and-SFX-Feature.jpg" style="center-align: middle;float:left;width: 110px;height: 100px" >
              <img src="https://www.wyzowl.com/wp-content/uploads/2019/01/The-20-Best-Royalty-Free-Music-Sites-in-2019.png" style="center-align: middle;float:left;width: 150px;height: 100px" >
              <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTBGSxAKPcy_oCxpYkHUcc4cNKPF-bEK6kIyQ&usqp=CAU" style="center-align: middle;float:right;width: 102px;height: 100px" >
            
	      """)
        st.subheader("Music recommendation system (Vibe.in)")
        st.write("A recommender system seeks to estimate and predict usercontent preference. The system draws from data usage history,aiming at making suggestions based on the user’s (current)interests.")
        st.write("There are two main types of recommender systems:")
        st.write("Content-based filters")
        st.write("Collaborative filters")
        st.write("Content-based filters predicts what a user likes based on what that particular user has liked in the past. On the other hand, collaborative-based filters predict what a user like based on what other users, that are similar to that particular user, have liked.")
        st.subheader("Collaborative filters")
        st.write("Collaborative Filters work with an interaction matrix, also called a rating matrix. The aim of this algorithm is to learn a function that can predict if a user will benefit from an item - meaning the user will likely buy, listen to, watch this item.Among collaborative-based systems, there can be two types: user-item filtering and item-item filtering.")
        st.write("What algorithms do collaborative filters use to recommend new songs?")
        st.write("There are several machine learning algorithms that can be used in the case of collaborative filtering. Among them, we can mention nearest-neighbor, clustering, and matrix factorization.")
        st.write("K-Nearest Neighbors (kNN) is considered the standard method when it comes to both user-based and item-based collaborative filtering approaches. In this app, we'll go through the steps for generating a music recommender system using a k-nearest algorithm approach.The aim is to build a collaborative filtering music recommender system using the Million Song Dataset; a freely available collection of audio features and metadata for a million contemporary popular music tracks.")
                

if __name__ == '__main__':
	main()

