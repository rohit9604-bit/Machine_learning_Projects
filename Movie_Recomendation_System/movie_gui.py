import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("900x650")
        self.root.resizable(True, True)
        
        # Load dataset
        try:
            csv_path = r'C:\Users\rohit\OneDrive\Desktop\mach_learning\Movie_Recomendation_System\movies.csv'
            self.movies_dataset = pd.read_csv(csv_path)
            self.vectorizer = TfidfVectorizer()
            self.prepare_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            return
        
        self.setup_ui()
    
    def prepare_model(self):
        """Prepare TF-IDF vectors and similarity matrix"""
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        for feature in selected_features:
            self.movies_dataset[feature] = self.movies_dataset[feature].fillna('')
        
        combined_features = (self.movies_dataset['genres'] + ' ' + 
                           self.movies_dataset['keywords'] + ' ' + 
                           self.movies_dataset['tagline'] + ' ' + 
                           self.movies_dataset['cast'] + ' ' + 
                           self.movies_dataset['director'])
        
        feature_vectors = self.vectorizer.fit_transform(combined_features)
        self.similarity = cosine_similarity(feature_vectors)
    
    def setup_ui(self):
        """Create GUI components"""
        # Title
        title_label = ttk.Label(self.root, text="🎬 Movie Recommendation System", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Search Movie", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Enter movie name:").pack(side=tk.LEFT, padx=5)
        
        self.movie_entry = ttk.Entry(input_frame, width=40, font=("Arial", 11))
        self.movie_entry.pack(side=tk.LEFT, padx=5)
        self.movie_entry.bind("<Return>", lambda e: self.get_recommendations())
        
        search_btn = ttk.Button(input_frame, text="Search", command=self.get_recommendations)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Recommendations", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matched movie display
        match_frame = ttk.Frame(results_frame)
        match_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(match_frame, text="Matched Movie:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.matched_label = ttk.Label(match_frame, text="", font=("Arial", 10, "italic"), 
                                       foreground="blue")
        self.matched_label.pack(side=tk.LEFT, padx=10)
        
        # Recommendations list
        ttk.Label(results_frame, text="Top 30 Recommendations:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=18, width=100, 
                                                      font=("Courier", 9), wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.config(state=tk.DISABLED)
        
        # Footer
        footer_label = ttk.Label(self.root, text="Press Enter or click Search to find recommendations", 
                                font=("Arial", 9, "italic"), foreground="gray")
        footer_label.pack(pady=5)
    
    def get_recommendations(self):
        """Fetch and display recommendations"""
        movie_name = self.movie_entry.get().strip()
        
        if not movie_name:
            messagebox.showwarning("Input Error", "Please enter a movie name")
            return
        
        try:
            titles = self.movies_dataset['title'].tolist()
            matches = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.6)
            
            if not matches:
                messagebox.showinfo("No Match", f"No close match found for '{movie_name}'.\nTry a different spelling.")
                self.matched_label.config(text="No match found")
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.config(state=tk.DISABLED)
                return
            
            matched_movie = matches[0]
            self.matched_label.config(text=matched_movie)
            
            # Get index and compute recommendations
            idx = self.movies_dataset[self.movies_dataset['title'] == matched_movie].index[0]
            rec_idx = self.similarity[idx].argsort()[::-1]
            
            recommendations = []
            for i in rec_idx:
                if i != idx:
                    recommendations.append(self.movies_dataset.loc[i, 'title'])
                if len(recommendations) >= 30:
                    break
            
            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            for num, title in enumerate(recommendations, 1):
                self.results_text.insert(tk.END, f"{num:2d}. {title}\n")
            
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommenderGUI(root)
    root.mainloop()
