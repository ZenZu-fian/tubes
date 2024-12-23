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

# Iterative Quick Sort function
def iterative_quick_sort(arr):
    stack = [(0, len(arr) - 1)]  # Stack to hold the start and end indices of sub-arrays

    while stack:
        start, end = stack.pop()
        
        if start < end:
            pivot_index = partition(arr, start, end)
            # Push the indices of the left and right sub-arrays onto the stack
            stack.append((start, pivot_index - 1))
            stack.append((pivot_index + 1, end))

    return arr

def partition(arr, start, end):
    pivot = arr[end]  # Choose the last element as the pivot
    i = start - 1     # Pointer for the smaller element

    for j in range(start, end):
        if arr[j] >= pivot:  # Change < to >= for descending order
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[end] = arr[end], arr[i + 1]  # Swap the pivot element with the element at i + 1
    return i + 1  # Return the index of the pivot

# Recursive Bubble Sort function (Descending Order)
def bubble_sort(arr, n=None):
    if n is None:
        n = len(arr)
    
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

# Iterative Bubble Sort function (Descending Order)
def iterative_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:  # Change > to < for descending order
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Recursive Insertion Sort function (Descending Order)
def insertion_sort(arr, n=None):
    if n is None:
        n = len(arr)
    
    # Base case: If the array size is 1 or less, return
    if n <= 1:
        return arr
    else:
        # Sort the first n-1 elements
        insertion_sort(arr, n - 1)
        
        # Insert the last element at the correct position
        key = arr[n - 1]
        j = n - 2
        
        # Change > to < for descending order
        while j >= 0 and arr[j] < key:  
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        
        return arr

# Iterative Insertion Sort function (Descending Order)
def iterative_insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        # Change > to < for descending order
        while j >= 0 and arr[j] < key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Measure sorting time for Quick Sort (Recursive)
start_time_quick = time.time()
sorted_average_scores_quick = average_scores.copy()
sorted_average_scores_quick['Average_UH'] = quick_sort(sorted_average_scores_quick['Average_UH'].tolist())
end_time_quick = time.time()
sorting_time_quick = end_time_quick - start_time_quick

# Measure sorting time for Iterative Quick Sort
start_time_iterative_quick = time.time()
sorted_average_scores_iterative_quick = average_scores.copy()
sorted_average_scores_iterative_quick['Average_UH'] = iterative_quick_sort(sorted_average_scores_iterative_quick['Average_UH'].tolist())
end_time_iterative_quick = time.time()
sorting_time_iterative_quick = end_time_iterative_quick - start_time_iterative_quick

# Measure sorting time for Bubble Sort (Recursive)
start_time_bubble = time.time()
sorted_average_scores_bubble = average_scores.copy()
sorted_average_scores_bubble['Average_UH'] = bubble_sort(sorted_average_scores_bubble['Average_UH'].tolist())
end_time_bubble = time.time()
sorting_time_bubble = end_time_bubble - start_time_bubble

# Measure sorting time for Iterative Bubble Sort
start_time_iterative_bubble = time.time()
sorted_average_scores_iterative_bubble = average_scores.copy()
sorted_average_scores_iterative_bubble['Average_UH'] = iterative_bubble_sort(sorted_average_scores_iterative_bubble['Average_UH'].tolist())
end_time_iterative_bubble = time.time()
sorting_time_iterative_bubble = end_time_iterative_bubble - start_time_iterative_bubble

# Measure sorting time for Insertion Sort (Recursive)
start_time_insertion = time.time()
sorted_average_scores_insertion = average_scores.copy()
sorted_average_scores_insertion['Average_UH'] = insertion_sort(sorted_average_scores_insertion['Average_UH'].tolist())
end_time_insertion = time.time()
sorting_time_insertion = end_time_insertion - start_time_insertion

# Measure sorting time for Iterative Insertion Sort
start_time_iterative_insertion = time.time()
sorted_average_scores_iterative_insertion = average_scores.copy()
sorted_average_scores_iterative_insertion['Average_UH'] = iterative_insertion_sort(sorted_average_scores_iterative_insertion['Average_UH'].tolist())
end_time_iterative_insertion = time.time()
sorting_time_iterative_insertion = end_time_iterative_insertion - start_time_iterative_insertion

# Display the sorted results
st.subheader("Sorted Results using Quick Sort (Recursive)")
st.dataframe(sorted_average_scores_quick)

st.subheader("Sorted Results using Quick Sort (Iterative)")
st.dataframe(sorted_average_scores_iterative_quick)

st.subheader("Sorted Results using Bubble Sort (Recursive)")
st.dataframe(sorted_average_scores_bubble)

st.subheader("Sorted Results using Bubble Sort (Iterative)")
st.dataframe(sorted_average_scores_iterative_bubble)

st.subheader("Sorted Results using Insertion Sort (Recursive)")
st.dataframe(sorted_average_scores_insertion)

st.subheader("Sorted Results using Insertion Sort (Iterative)")
st.dataframe(sorted_average_scores_iterative_insertion)

# Display the sorting times
st.subheader("Time taken for each sorting")
st.write(f"Time taken to sort using Quick Sort (Recursive): {sorting_time_quick:.6f} seconds")
st.write(f"Time taken to sort using Quick Sort (Iterative): {sorting_time_iterative_quick:.6f} seconds")
st.write(f"Time taken to sort using Bubble Sort (Recursive): {sorting_time_bubble:.6f} seconds")
st.write(f"Time taken to sort using Bubble Sort (Iterative): {sorting_time_iterative_bubble:.6f} seconds")
st.write(f"Time taken to sort using Insertion Sort (Recursive): {sorting_time_insertion:.6f} seconds")
st.write(f"Time taken to sort using Insertion Sort (Iterative): {sorting_time_iterative_insertion:.6f} seconds")

# Plotting the sorting times for iterative algorithms
st.subheader("Iterative Sorting Time Visualization")
fig_iterative, ax_iterative = plt.subplots()
ax_iterative.barh(['Quick Sort (Iterative)', 'Bubble Sort (Iterative)', 'Insertion Sort (Iterative)'], 
                   [sorting_time_iterative_quick, sorting_time_iterative_bubble, sorting_time_iterative_insertion], 
                   color=['lightgreen', 'orange', 'cyan'])
ax_iterative.set_xlabel('Time (seconds)')
ax_iterative.set_title('Time taken for Iterative Sorting Algorithms')
st.pyplot(fig_iterative)

# Plot ting the sorting times for recursive algorithms
st.subheader("Recursive Sorting Time Visualization")
fig_recursive, ax_recursive = plt.subplots()
ax_recursive.barh(['Quick Sort (Recursive)', 'Bubble Sort (Recursive)', 'Insertion Sort (Recursive)'], 
                   [sorting_time_quick, sorting_time_bubble, sorting_time_insertion], 
                   color=['skyblue', 'orange', 'purple'])
ax_recursive.set_xlabel('Time (seconds)')
ax_recursive.set_title('Time taken for Recursive Sorting Algorithms')
st.pyplot(fig_recursive)

# Plotting the sorting times for all algorithms
st.subheader("All Sorting Time Visualization")
fig_all, ax_all = plt.subplots()
ax_all.barh(['Quick Sort (Recursive)', 'Quick Sort (Iterative)', 'Bubble Sort (Recursive)', 
              'Bubble Sort (Iterative)', 'Insertion Sort (Recursive)', 'Insertion Sort (Iterative)'], 
             [sorting_time_quick, sorting_time_iterative_quick, sorting_time_bubble, 
              sorting_time_iterative_bubble, sorting_time_insertion, sorting_time_iterative_insertion], 
             color=['skyblue', 'lightgreen', 'orange', 'red', 'purple', 'cyan'])
ax_all.set_xlabel('Time (seconds)')
ax_all.set_title('Time taken for All Sorting Algorithms')
st.pyplot(fig_all)