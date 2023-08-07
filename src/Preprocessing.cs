using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    public enum State { 
        PuertoRico, VirginIslands, Massachusetts, Connecticut, NewHampshire, Vermont, NewJersey, NewYork, SouthCarolina, 
        Tennessee, RhodeIsland, Virginia, Wyoming, Maine, Georgia, Pennsylvania, WestVirginia, Delaware 
    }

    public class Property {
        public double? price { get; set; }
        public double? bed { get; set; }
        public double? bath { get; set; }
        public double? acre_lot { get; set; }
        public State? state { get; set; }
        public double? zip_code { get; set; }
        public double? house_size { get; set; }

        [Ignore]
        public int? propertyClass { get; set; }
    }

    public class Preprocessing {
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

        public static int UniqueClasses = 5;

        public static List<Property> LoadAndPreprocessData() {
            List<Property> properties;
            using (var reader = new StreamReader("realtor-data.csv"))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
                csv.Context.TypeConverterCache.AddConverter<State>(new StateEnumConverter());
                properties = csv.GetRecords<Property>().ToList();
            }

            properties = FilterByPercentile(properties);
            FillMissingWithAverage(properties, p => p.zip_code, (p, value) => p.zip_code = value);

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

            return Standardize(properties);
        }

        public static List<Property> FilterByPercentile(List<Property> properties) {
            var priceTask = Task.Run(() => GetPercentile(properties.Where(p => p.price.HasValue).Select(p => p.price.Value), 0.95));
            var bedTask = Task.Run(() => GetPercentile(properties.Where(p => p.bed.HasValue).Select(p => p.bed.Value), 0.95));
            var bathTask = Task.Run(() => GetPercentile(properties.Where(p => p.bath.HasValue).Select(p => p.bath.Value), 0.95));
            var acreLotTask = Task.Run(() => GetPercentile(properties.Where(p => p.acre_lot.HasValue).Select(p => p.acre_lot.Value), 0.95));
            var houseSizeTask = Task.Run(() => GetPercentile(properties.Where(p => p.house_size.HasValue).Select(p => p.house_size.Value), 0.95));

            Task.WaitAll(priceTask, bedTask, bathTask, acreLotTask, houseSizeTask);

            var price_percentile = priceTask.Result;
            var bed_percentile = bedTask.Result;
            var bath_percentile = bathTask.Result;
            var acre_lot_percentile = acreLotTask.Result;
            var house_size_percentile = houseSizeTask.Result;

            return properties.AsParallel().Where(p =>
                p.price.HasValue && p.price.Value <= price_percentile && p.price.Value > 1.0 &&
                p.bed.HasValue && p.bed.Value <= bed_percentile &&
                p.bath.HasValue && p.bath.Value <= bath_percentile &&
                p.acre_lot.HasValue && p.acre_lot.Value <= acre_lot_percentile &&
                p.house_size.HasValue && p.house_size.Value <= house_size_percentile).ToList();
        }

        public static double GetPercentile(IEnumerable<double> sequence, double percentile) {
            var orderedSequence = sequence.OrderBy(x => x).ToList();
            int N = orderedSequence.Count;
            double n = (N - 1) * percentile + 1;
            
            int k = (int)Math.Floor(n);
            
            if (n == k) {
                return orderedSequence[k - 1];
            }

            double d = n - k;
            return orderedSequence[k - 1] + d * (orderedSequence[k] - orderedSequence[k - 1]);
        }

        public static void FillMissingWithAverage<T>(List<T> items, Func<T, double?> selector, Action<T, double> setter) {
            if (items.Any(item => selector(item) == null)) {
                var average = items.Where(item => selector(item) != null).Average(selector);
                
                Parallel.ForEach(items.Where(item => selector(item) == null), item => {
                    setter(item, average.Value);
                });
            }
        }

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

            Parallel.ForEach(properties, property => {
                property.bed = (bed_std == 0) ? 0 : (property.bed.Value - bed_mean) / bed_std;
                property.bath = (bath_std == 0) ? 0 : (property.bath.Value - bath_mean) / bath_std;
                property.acre_lot = (acre_lot_std == 0) ? 0 : (property.acre_lot.Value - acre_lot_mean) / acre_lot_std;
                property.zip_code = (zip_code_std == 0) ? 0 : (property.zip_code.Value - zip_code_mean) / zip_code_std;
                property.house_size = (house_size_std == 0) ? 0 : (property.house_size.Value - house_size_mean) / house_size_std;
            });

            return properties;
        }



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

        public static List<double> OneHotEncodeState(State state) {
            var encoding = new double[Enum.GetValues(typeof(State)).Length];
            encoding[(int)state] = 1;
            return encoding.ToList();
        }
    }
}
