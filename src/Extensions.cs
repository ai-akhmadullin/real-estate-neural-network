namespace RealEstate {
    /// <summary>
    /// Provides extension methods for enumerable collections.
    /// </summary>
    public static class IEnumerableExtensions {
        /// <summary>
        /// Divides the source enumerable collection into batches of a specified size.
        /// </summary>
        /// <typeparam name="T">The type of the elements in the source collection.</typeparam>
        /// <param name="source">The source enumerable collection.</param>
        /// <param name="size">The size of each batch.</param>
        /// <returns>An enumerable collection of batches, where each batch is represented as an enumerable collection of elements.</returns>
        public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size) {
            // Check for negative or zero size.
            if (size <= 0) {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            
            T[] bucket = null;
            var count = 0;
            
            // Iterate through the source collection.
            foreach (var item in source) {
                // Initialize a new bucket if it's null.
                if (bucket == null) bucket = new T[size]; 
                
                // Add the current item to the bucket.
                bucket[count++] = item;

                // If the bucket is not full, continue to the next item.
                if (count != size) continue;

                // Yield the full bucket and reset the bucket and count.
                yield return bucket;

                bucket = null;
                count = 0;
            }
            
            // If there are any remaining items in the last bucket, yield them.
            if (bucket != null && count > 0) {
                yield return bucket.Take(count);
            }
        }
    }
}