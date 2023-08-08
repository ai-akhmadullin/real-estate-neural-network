using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    /// <summary>
    /// Enumeration representing the available States.
    /// </summary>
    public enum State { 
        PuertoRico, VirginIslands, Massachusetts, Connecticut, NewHampshire, Vermont, NewJersey, NewYork, SouthCarolina, 
        Tennessee, RhodeIsland, Virginia, Wyoming, Maine, Georgia, Pennsylvania, WestVirginia, Delaware 
    }

    /// <summary>
    /// Class representing a property with various features.
    /// </summary>
    public class Property {
        public double? price { get; set; } // The price in US dollars.
        public double? bed { get; set; } // The number of bedrooms.
        public double? bath { get; set; } // The number of bathrooms.
        public double? acre_lot { get; set; } // Total property/land size in acres.
        public State? state { get; set; } // The state (e.g. Puerto Rico, Maine, Georgia) in which the property locates.
        public double? zip_code { get; set; } // Postal code of the area.
        public double? house_size { get; set; } // Building area/living space in square feet

        [Ignore]
        public int? propertyClass { get; set; } // Classifying property based on price range.
    }

    /// <summary>
    /// Class containing methods for preprocessing real estate data.
    /// </summary>
    public class Preprocessing {
        /// <summary>
        /// Converter to handle the State enumeration in CSV files.
        /// </summary>
        public class StateEnumConverter : CsvHelper.TypeConversion.DefaultTypeConverter {
            public override object? ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData) {
                if (string.IsNullOrEmpty(text)) {
                    return null;
                }
                return (State)Enum.Parse(typeof(State), text.Replace(" ", ""));
            }

            public override string? ConvertToString(object value, IWriterRow row, MemberMapData memberMapData) {
                return value?.ToString();
            }
        }

        public static int UniqueClasses = 5; // Number of unique property classes, with 1 as the lowest and `UniqueClasses` as the highest (with the highest price) class.

        /// <summary>
        /// Loads and preprocesses real estate data from a CSV file.
        /// </summary>
        /// <returns>A list of preprocessed properties.</returns>
        public static List<Property> LoadAndPreprocessData() {
            List<Property> properties;
            
            try {
                using (var reader = new StreamReader("realtor-data.csv"))
                using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
                    csv.Context.TypeConverterCache.AddConverter<State>(new StateEnumConverter());
                    properties = csv.GetRecords<Property>().ToList();
                }
            } catch (Exception e) {
                throw new InvalidOperationException("Error reading CSV file.", e);
            }

            // Keep only those properties whose features are below the 95th percentile, thus removing the properties with peaked features.
            properties = FilterByPercentile(properties, 0.95).Result;

            // For each property if the value of the feature is null, replace it with an average value.
            FillMissingWithAverage(properties, p => p.zip_code, (p, value) => p.zip_code = value);

            // Classify properties based on the price range.
            double[] prices = properties.Select(p => p.price.Value).ToArray();
            double minPriceThreshold = GetPercentile(prices, 0.2);
            double maxPriceThreshold = GetPercentile(prices, 0.8);

            double range = (maxPriceThreshold - minPriceThreshold) / (UniqueClasses - 2);

            Parallel.ForEach(properties, property => {
                double price = property.price.Value;

                if (price <= minPriceThreshold) property.propertyClass = 1;
                else if (price >= maxPriceThreshold) property.propertyClass = UniqueClasses;
                else {
                    for (int i = 0; i < UniqueClasses - 2; i++) {
                        if (price > minPriceThreshold + i * range && price <= minPriceThreshold + (i + 1) * range) {
                            property.propertyClass = i + 2;
                            break;
                        }
                    }
                }
            });

            // Apply z-score normalization on the data before returning it.
            return Standardize(properties, true);
        }

        /// <summary>
        /// Filters properties by the `percentile`th percentile for each feature.
        /// </summary>
        /// <param name="properties">List of properties to filter.</param>
        /// <returns>Filtered list of properties.</returns>
        public static async Task<List<Property>> FilterByPercentile(List<Property> properties, double percentile) {
            var priceTask = Task.Run(() => GetPercentile(properties.Where(p => p.price.HasValue).Select(p => p.price.Value), percentile));
            var bedTask = Task.Run(() => GetPercentile(properties.Where(p => p.bed.HasValue).Select(p => p.bed.Value), percentile));
            var bathTask = Task.Run(() => GetPercentile(properties.Where(p => p.bath.HasValue).Select(p => p.bath.Value), percentile));
            var acreLotTask = Task.Run(() => GetPercentile(properties.Where(p => p.acre_lot.HasValue).Select(p => p.acre_lot.Value), percentile));
            var houseSizeTask = Task.Run(() => GetPercentile(properties.Where(p => p.house_size.HasValue).Select(p => p.house_size.Value), percentile));

            var price_percentile = await priceTask;
            var bed_percentile = await bedTask;
            var bath_percentile = await bathTask;
            var acre_lot_percentile = await acreLotTask;
            var house_size_percentile = await houseSizeTask;
            
            // Return only those properties whose features are below the percentile.
            return properties.AsParallel().Where(p =>
                p.price.HasValue && p.price.Value <= price_percentile && p.price.Value > 1.0 &&
                p.bed.HasValue && p.bed.Value <= bed_percentile &&
                p.bath.HasValue && p.bath.Value <= bath_percentile &&
                p.acre_lot.HasValue && p.acre_lot.Value <= acre_lot_percentile &&
                p.house_size.HasValue && p.house_size.Value <= house_size_percentile).ToList();
        }


        /// <summary>
        /// Computes the percentile value of a given sequence.
        /// </summary>
        /// <param name="sequence">Sequence of values.</param>
        /// <param name="percentile">Percentile to compute.</param>
        /// <returns>The value of the specified percentile.</returns>
        public static double GetPercentile(IEnumerable<double> sequence, double percentile) {
            var orderedSequence = sequence.OrderBy(x => x).ToList();
            int N = orderedSequence.Count;
            double n = (N - 1) * percentile + 1;
            
            int k = (int)Math.Floor(n);
            
            // If n is an integer, return the value at the index `n-1`.
            if (n == k) {
                return orderedSequence[k - 1];
            }

            // Otherwise, compute the interpolated value.
            double d = n - k;
            return orderedSequence[k - 1] + d * (orderedSequence[k] - orderedSequence[k - 1]);
        }

        /// <summary>
        /// Fills missing values in a list with an average value.
        /// </summary>
        /// <typeparam name="T">Type of the items in the list.</typeparam>
        /// <param name="items">List of items.</param>
        /// <param name="selector">Function to select the value to check.</param>
        /// <param name="setter">Action to set the value.</param>
        public static void FillMissingWithAverage<T>(List<T> items, Func<T, double?> selector, Action<T, double> setter) {
            if (items.Any(item => selector(item) == null)) {
                var average = items.Where(item => selector(item) != null).Average(selector);
                
                Parallel.ForEach(items.Where(item => selector(item) == null), item => {
                    setter(item, average.Value);
                });
            }
        }

        /// <summary>
        /// Standardizes the properties using z-score normalization.
        /// </summary>
        /// <param name="properties">List of properties to standardize.</param>
        /// <returns>Standardized list of properties.</returns>
        public static List<Property> Standardize(List<Property> properties) {
            double bed_mean = 0, bed_std = 0, bath_mean = 0, bath_std = 0, acre_lot_mean = 0, acre_lot_std = 0,
                    zip_code_mean = 0, zip_code_std = 0, house_size_mean = 0, house_size_std = 0;

            Parallel.Invoke(
                () => { bed_mean = properties.Average(p => p.bed.Value); bed_std = Math.Sqrt(properties.Average(p => Math.Pow(p.bed.Value - bed_mean, 2))); },
                () => { bath_mean = properties.Average(p => p.bath.Value); bath_std = Math.Sqrt(properties.Average(p => Math.Pow(p.bath.Value - bath_mean, 2))); },
                () => { acre_lot_mean = properties.Average(p => p.acre_lot.Value); acre_lot_std = Math.Sqrt(properties.Average(p => Math.Pow(p.acre_lot.Value - acre_lot_mean, 2))); },
                () => { zip_code_mean = properties.Average(p => p.zip_code.Value); zip_code_std = Math.Sqrt(properties.Average(p => Math.Pow(p.zip_code.Value - zip_code_mean, 2))); },
                () => { house_size_mean = properties.Average(p => p.house_size.Value); house_size_std = Math.Sqrt(properties.Average(p => Math.Pow(p.house_size.Value - house_size_mean, 2))); }
            );

            // New value = (x – μ) / σ  ,  where:
            // x: Original value
            // μ: Mean of data
            // σ: Standard deviation of data
            Parallel.ForEach(properties, property => {
                property.bed = (bed_std == 0) ? 0 : (property.bed.Value - bed_mean) / bed_std;
                property.bath = (bath_std == 0) ? 0 : (property.bath.Value - bath_mean) / bath_std;
                property.acre_lot = (acre_lot_std == 0) ? 0 : (property.acre_lot.Value - acre_lot_mean) / acre_lot_std;
                property.zip_code = (zip_code_std == 0) ? 0 : (property.zip_code.Value - zip_code_mean) / zip_code_std;
                property.house_size = (house_size_std == 0) ? 0 : (property.house_size.Value - house_size_mean) / house_size_std;
            });

            return properties;
        }

        /// <summary>
        /// Standardizes the properties using either z-score normalization or min-max scaling.
        /// </summary>
        /// <param name="properties">List of properties to standardize.</param>
        /// <param name="useZScore">If true, uses z-score normalization. If false, uses min-max scaling.</param>
        /// <returns>Standardized list of properties.</returns>
        public static List<Property> Standardize(List<Property> properties, bool useZScore = true) {
            // Z-score normalization.
            if (useZScore) {
                double bed_mean = 0, bed_std = 0, bath_mean = 0, bath_std = 0, acre_lot_mean = 0, acre_lot_std = 0,
                        zip_code_mean = 0, zip_code_std = 0, house_size_mean = 0, house_size_std = 0;

                Parallel.Invoke(
                    () => { bed_mean = properties.Average(p => p.bed.Value); bed_std = Math.Sqrt(properties.Average(p => Math.Pow(p.bed.Value - bed_mean, 2))); },
                    () => { bath_mean = properties.Average(p => p.bath.Value); bath_std = Math.Sqrt(properties.Average(p => Math.Pow(p.bath.Value - bath_mean, 2))); },
                    () => { acre_lot_mean = properties.Average(p => p.acre_lot.Value); acre_lot_std = Math.Sqrt(properties.Average(p => Math.Pow(p.acre_lot.Value - acre_lot_mean, 2))); },
                    () => { zip_code_mean = properties.Average(p => p.zip_code.Value); zip_code_std = Math.Sqrt(properties.Average(p => Math.Pow(p.zip_code.Value - zip_code_mean, 2))); },
                    () => { house_size_mean = properties.Average(p => p.house_size.Value); house_size_std = Math.Sqrt(properties.Average(p => Math.Pow(p.house_size.Value - house_size_mean, 2))); }
                );

                // New value = (x – μ) / σ  ,  where:
                // x: Original value
                // μ: Mean of data
                // σ: Standard deviation of data
                Parallel.ForEach(properties, property => {
                    property.bed = (bed_std == 0) ? 0 : (property.bed.Value - bed_mean) / bed_std;
                    property.bath = (bath_std == 0) ? 0 : (property.bath.Value - bath_mean) / bath_std;
                    property.acre_lot = (acre_lot_std == 0) ? 0 : (property.acre_lot.Value - acre_lot_mean) / acre_lot_std;
                    property.zip_code = (zip_code_std == 0) ? 0 : (property.zip_code.Value - zip_code_mean) / zip_code_std;
                    property.house_size = (house_size_std == 0) ? 0 : (property.house_size.Value - house_size_mean) / house_size_std;
                });
            } 

            // Min-max scaling
            else {
                // Define minimum and maximum values for each feature
                double bed_min = properties.Min(p => p.bed.Value), bed_max = properties.Max(p => p.bed.Value);
                double bath_min = properties.Min(p => p.bath.Value), bath_max = properties.Max(p => p.bath.Value);
                double acre_lot_min = properties.Min(p => p.acre_lot.Value), acre_lot_max = properties.Max(p => p.acre_lot.Value);
                double zip_code_min = properties.Min(p => p.zip_code.Value), zip_code_max = properties.Max(p => p.zip_code.Value);
                double house_size_min = properties.Min(p => p.house_size.Value), house_size_max = properties.Max(p => p.house_size.Value);

                // New value = (x - min) / (max - min)
                Parallel.ForEach(properties, property => {
                    property.bed = (bed_max - bed_min == 0) ? 0 : (property.bed.Value - bed_min) / (bed_max - bed_min);
                    property.bath = (bath_max - bath_min == 0) ? 0 : (property.bath.Value - bath_min) / (bath_max - bath_min);
                    property.acre_lot = (acre_lot_max - acre_lot_min == 0) ? 0 : (property.acre_lot.Value - acre_lot_min) / (acre_lot_max - acre_lot_min);
                    property.zip_code = (zip_code_max - zip_code_min == 0) ? 0 : (property.zip_code.Value - zip_code_min) / (zip_code_max - zip_code_min);
                    property.house_size = (house_size_max - house_size_min == 0) ? 0 : (property.house_size.Value - house_size_min) / (house_size_max - house_size_min);
                });
            }

            return properties;
        }


        /// <summary>
        /// Splits data into train and test sets.
        /// </summary>
        /// <param name="data">List of properties to split.</param>
        /// <param name="trainSizeRatio">Proportion of the data to use for training.</param>
        /// <returns>Tuple of train and test data and targets.</returns>
        public static (List<Vector<double>>, List<int?>, List<Vector<double>>, List<int?>) SplitData(List<Property> data, double trainSizeRatio) {
            int trainCount = (int)(data.Count * trainSizeRatio);
            var rng = new Random();
            var shuffledData = data.OrderBy(x => rng.Next()).ToList();

            var trainData = shuffledData.Take(trainCount).Select(p => PropertyToVector(p)).ToList();
            var trainTargets = shuffledData.Take(trainCount).Select(p => p.propertyClass).ToList();
            var testData = shuffledData.Skip(trainCount).Select(p => PropertyToVector(p)).ToList();
            var testTargets = shuffledData.Skip(trainCount).Select(p => p.propertyClass).ToList();

            return (trainData, trainTargets, testData, testTargets);
        }

        /// <summary>
        /// Converts a Property object to a feature vector.
        /// </summary>
        /// <param name="property">Property object to convert.</param>
        /// <returns>Feature vector representation of the property.</returns>
        public static Vector<double> PropertyToVector(Property property) {
             var stateOneHot = OneHotEncodeState(property.state ?? State.PuertoRico);
             var features = new List<double> { 
                property.bed ?? 0,
                property.bath ?? 0,
                property.acre_lot ?? 0,
                property.zip_code ?? 0,
                property.house_size ?? 0
            };
            features.AddRange(stateOneHot);

            return Vector<double>.Build.DenseOfEnumerable(features);
        }

        /// <summary>
        /// One-hot encodes a given State.
        /// </summary>
        /// <param name="state">State to encode.</param>
        /// <returns>One-hot encoded representation of the State.</returns>
        public static List<double> OneHotEncodeState(State state) {
            var encoding = new double[Enum.GetValues(typeof(State)).Length];
            encoding[(int)state] = 1;
            return encoding.ToList();
        }
    }
}
