import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel('Data-UH.xlsx')

# Assuming the columns are named 'Nama siswa', 'Kelas', 'UH1', 'UH2', 'UH3'
average_scores = df[['Nama siswa', 'Kelas', 'UH1', 'UH2', 'UH3']].copy()

# Calculate the average for UH1, UH2, and UH3
average_scores['Average_UH'] = average_scores[['UH1', 'UH2', 'UH3']].mean(axis=1)

# Display the results
st.dataframe(average_scores)

# Recursive Quick Sort function (Descending Order)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x > pivot]  # Change < to > for descending order
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x < pivot]  # Change > to < for descending order
        return quick_sort(left) + middle + quick_sort(right)

# Recursive Bubble Sort function (Descending Order)
def bubble_sort(arr, n=345):
    
    # Base case: If the array size is 1, return
    if n == 1:
        return arr
    else:
        # One pass of bubble sort (Descending Order)
        for i in range(n - 1):
            if arr[i] < arr[i + 1]:  # Change > to < for descending order
                arr[i], arr[i + 1] = arr[i + 1], arr[i]

        # Recursive call for the remaining elements
        return bubble_sort(arr, n - 1)

# Iterative Insertion Sort function (Descending Order)
def iterative_insertion_sort(arr):
    n = len(arr)
    i = 1
    while i < n:
        key = arr[i]
        j = i - 1
        
        # Change > to < for descending order
        while j >= 0 and arr[j] < key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        i += 1
    return arr

# Measure sorting time for Quick Sort (Recursive)
start_time_quick = time.time()
sorted_average_scores_quick = average_scores.copy()
sorted_average_scores_quick['Average_UH'] = quick_sort(sorted_average_scores_quick['Average_UH'].tolist())
end_time_quick = time.time()
sorting_time_quick = end_time_quick - start_time_quick

# Measure sorting time for Bubble Sort (Recursive)
start_time_bubble = time.time()
sorted_average_scores_bubble = average_scores.copy()
sorted_average_scores_bubble['Average_UH'] = bubble_sort(sorted_average_scores_bubble['Average_UH'].tolist())
end_time_bubble = time.time()
sorting_time_bubble = end_time_bubble - start_time_bubble

# Measure sorting time for Iterative Insertion Sort
start_time_iterative_insertion = time.time()
sorted_average_scores_iterative_insertion = average_scores.copy()
sorted_average_scores_iterative_insertion['Average_UH'] = iterative_insertion_sort(sorted_average_scores_iterative_insertion['Average_UH'].tolist())
end_time_iterative_insertion = time.time()
sorting_time_iterative_insertion = end_time_iterative_insertion - start_time_iterative_insertion

# Display the sorted results
st.subheader("Sorted Results using Quick Sort (Recursive)")
st.dataframe(sorted_average_scores_quick)

st.subheader("Sorted Results using Bubble Sort (Recursive)")
st.dataframe(sorted_average_scores_bubble)

st.subheader("Sorted Results using Insertion Sort (Iterative)")
st.dataframe(sorted_average_scores_iterative_insertion)

# Display the sorting times
st.subheader("Time taken for each sorting")
st.write(f"Time taken to sort using Quick Sort (Recursive): {sorting_time_quick:.6f} seconds")
st.write(f"Time taken to sort using Bubble Sort (Recursive): {sorting_time_bubble:.6f} seconds")
st.write(f"Time taken to sort using Insertion Sort (Iterative): {sorting_time_iterative_insertion:.6f} seconds")

# Plotting the sorting times for all algorithms
st.subheader("All Sorting Time Visualization")
fig_all, ax_all = plt.subplots()
ax_all.barh(['Quick Sort (Recursive)', 'Bubble Sort (Recursive)', 'Insertion Sort (Iterative)'], 
             [sorting_time_quick, sorting_time_bubble, sorting_time_iterative_insertion], 
             color=['skyblue', 'red', 'cyan'])
ax_all.set_xlabel('Time (seconds)')
ax_all.set_title('Time taken for Selected Sorting Algorithms')
st.pyplot(fig_all)