using System;
using System.IO;
using System.Security.Cryptography;
using System.Threading;

public static class ZMath
{
    // Базовые математические константы
    public static readonly double PI = Math.PI;
    public static readonly double E = Math.E;
    public static readonly double GOLDEN_RATIO = (1 + Math.Sqrt(5)) / 2;

    // Константы Фейгенбаума
    public static readonly double Feigenbaum1 = 4.669;
    public static readonly double Feigenbaum2 = 2.5029;

    // Дополнительные константы
    public static readonly double Apery = 1.202056903159594;       // ζ(3)
    public static readonly double Plastic = 1.324717957244746;     // Пластическое число
    public static readonly double Sqrt2 = Math.Sqrt(2);            // √2
    public static readonly double Sqrt3 = Math.Sqrt(3);            // √3
    public static readonly double Sqrt5 = Math.Sqrt(5);            // √5
    public static readonly double Ln2 = Math.Log(2);               // ln(2)
    public static readonly double SuperGoldenRatio = 1.465571231876768; // Сверхзолотое сечение
    public static readonly double Gelfond = Math.Exp(PI);          // e^π

    // Потокобезопасные генераторы
    private static readonly ThreadLocal<Random> _random = new ThreadLocal<Random>(() =>
    {
        using (var rng = new RNGCryptoServiceProvider())
        {
            byte[] seedBytes = new byte[4];
            rng.GetBytes(seedBytes);
            return new Random(BitConverter.ToInt32(seedBytes, 0));
        }
    });

    private static readonly ThreadLocal<RNGCryptoServiceProvider> _cryptoRng =
        new ThreadLocal<RNGCryptoServiceProvider>(() => new RNGCryptoServiceProvider());

    // Случайная константа Фейгенбаума
    public static double Feigenbaum3D
    {
        get
        {
            var rnd = _random.Value;
            switch (rnd.Next(0, 3))
            {
                case 0: return (Feigenbaum1 + Feigenbaum2) / 2.0;
                case 1: return Feigenbaum1;
                case 2: return Feigenbaum2;
                default: return (Feigenbaum1 + Feigenbaum2) / 2.0;
            }
        }
    }

    // Криптостойкий генератор случайных целых
    public static double VeryRandom
    {
        get
        {
            byte[] bytes = new byte[4];
            _cryptoRng.Value.GetBytes(bytes);
            uint randomValue = BitConverter.ToUInt32(bytes, 0);
            return (double)(1 + (randomValue % (int.MaxValue - 1)));
        }
    }

    // Генерация комбинаций констант (более 200 вариантов)
    private static double Combine(Func<double, double, double> operation, double a, double b) => operation(a, b);

    public static double Add(double a, double b) => a + b;
    public static double Subtract(double a, double b) => a - b;
    public static double Multiply(double a, double b) => a * b;
    public static double Divide(double a, double b) => a / b;

    // Основные комбинации
    public static double Feigenbaum3 => Combine(Add, Feigenbaum1, Feigenbaum2) / 2.0;
    public static double Feigenbaum4 => Combine(Add, Feigenbaum1, Feigenbaum2) / PI;
    public static double Feigenbaum5 => Combine(Divide, Feigenbaum1, PI);
    public static double Feigenbaum6 => Combine(Divide, Feigenbaum2, PI);
    public static double Feigenbaum7 => Combine(Divide, PI, Feigenbaum1);
    public static double Feigenbaum8 => Combine(Divide, PI, Feigenbaum2);
    public static double Feigenbaum9 => Combine(Divide, GOLDEN_RATIO, PI);
    public static double Feigenbaum10 => Combine(Divide, PI, GOLDEN_RATIO);
    public static double Feigenbaum11 => Combine(Divide, E, PI);
    public static double Feigenbaum12 => Combine(Divide, PI, E);

    // Добавляем все комбинации для основных констант
    public static double Apery_div_PI => Combine(Divide, Apery, PI);
    public static double PI_div_Apery => Combine(Divide, PI, Apery);
    public static double Plastic_div_PI => Combine(Divide, Plastic, PI);
    public static double PI_div_Plastic => Combine(Divide, PI, Plastic);
    public static double Sqrt2_div_PI => Combine(Divide, Sqrt2, PI);
    public static double PI_div_Sqrt2 => Combine(Divide, PI, Sqrt2);
    public static double Sqrt3_div_PI => Combine(Divide, Sqrt3, PI);
    public static double PI_div_Sqrt3 => Combine(Divide, PI, Sqrt3);
    public static double Sqrt5_div_PI => Combine(Divide, Sqrt5, PI);
    public static double PI_div_Sqrt5 => Combine(Divide, PI, Sqrt5);
    public static double Ln2_div_PI => Combine(Divide, Ln2, PI);
    public static double PI_div_Ln2 => Combine(Divide, PI, Ln2);
    public static double SuperGoldenRatio_div_PI => Combine(Divide, SuperGoldenRatio, PI);
    public static double PI_div_SuperGoldenRatio => Combine(Divide, PI, SuperGoldenRatio);
    public static double Gelfond_div_PI => Combine(Divide, Gelfond, PI);
    public static double PI_div_Gelfond => Combine(Divide, PI, Gelfond);
    public static double GOLDEN_RATIO_div_PI => Combine(Divide, GOLDEN_RATIO, PI);
    public static double PI_div_GOLDEN_RATIO => Combine(Divide, PI, GOLDEN_RATIO);
    public static double PI_div_E => Combine(Divide, PI, E);

    // Комбинации с E
    public static double Apery_div_E => Combine(Divide, Apery, E);
    public static double E_div_Apery => Combine(Divide, E, Apery);
    public static double Plastic_div_E => Combine(Divide, Plastic, E);
    public static double E_div_Plastic => Combine(Divide, E, Plastic);
    public static double Sqrt2_div_E => Combine(Divide, Sqrt2, E);
    public static double E_div_Sqrt2 => Combine(Divide, E, Sqrt2);
    public static double Sqrt3_div_E => Combine(Divide, Sqrt3, E);
    public static double E_div_Sqrt3 => Combine(Divide, E, Sqrt3);
    public static double Sqrt5_div_E => Combine(Divide, Sqrt5, E);
    public static double E_div_Sqrt5 => Combine(Divide, E, Sqrt5);
    public static double Ln2_div_E => Combine(Divide, Ln2, E);
    public static double E_div_Ln2 => Combine(Divide, E, Ln2);
    public static double SuperGoldenRatio_div_E => Combine(Divide, SuperGoldenRatio, E);
    public static double E_div_SuperGoldenRatio => Combine(Divide, E, SuperGoldenRatio);
    public static double Gelfond_div_E => Combine(Divide, Gelfond, E);
    public static double E_div_Gelfond => Combine(Divide, E, Gelfond);
    public static double GOLDEN_RATIO_div_E => Combine(Divide, GOLDEN_RATIO, E);
    public static double E_div_GOLDEN_RATIO => Combine(Divide, E, GOLDEN_RATIO);
    public static double E_div_PI => Combine(Divide, E, PI);

    // Комбинации с золотым сечением
    public static double Apery_div_GOLDEN => Combine(Divide, Apery, GOLDEN_RATIO);
    public static double GOLDEN_div_Apery => Combine(Divide, GOLDEN_RATIO, Apery);
    public static double Plastic_div_GOLDEN => Combine(Divide, Plastic, GOLDEN_RATIO);
    public static double GOLDEN_div_Plastic => Combine(Divide, GOLDEN_RATIO, Plastic);
    public static double Sqrt2_div_GOLDEN => Combine(Divide, Sqrt2, GOLDEN_RATIO);
    public static double GOLDEN_div_Sqrt2 => Combine(Divide, GOLDEN_RATIO, Sqrt2);
    public static double Sqrt3_div_GOLDEN => Combine(Divide, Sqrt3, GOLDEN_RATIO);
    public static double GOLDEN_div_Sqrt3 => Combine(Divide, GOLDEN_RATIO, Sqrt3);
    public static double Sqrt5_div_GOLDEN => Combine(Divide, Sqrt5, GOLDEN_RATIO);
    public static double GOLDEN_div_Sqrt5 => Combine(Divide, GOLDEN_RATIO, Sqrt5);
    public static double Ln2_div_GOLDEN => Combine(Divide, Ln2, GOLDEN_RATIO);
    public static double GOLDEN_div_Ln2 => Combine(Divide, GOLDEN_RATIO, Ln2);
    public static double SuperGoldenRatio_div_GOLDEN => Combine(Divide, SuperGoldenRatio, GOLDEN_RATIO);
    public static double GOLDEN_div_SuperGoldenRatio => Combine(Divide, GOLDEN_RATIO, SuperGoldenRatio);
    public static double Gelfond_div_GOLDEN => Combine(Divide, Gelfond, GOLDEN_RATIO);
    public static double GOLDEN_div_Gelfond => Combine(Divide, GOLDEN_RATIO, Gelfond);
    public static double PI_div_GOLDEN => Combine(Divide, PI, GOLDEN_RATIO);
    public static double GOLDEN_div_PI => Combine(Divide, GOLDEN_RATIO, PI);
    public static double E_div_GOLDEN => Combine(Divide, E, GOLDEN_RATIO);
    public static double GOLDEN_div_E => Combine(Divide, GOLDEN_RATIO, E);

    // Динамический выбор функции через конфиг-файл
    private static readonly object _configLock = new object();
    private static string _cachedFuncName;
    private static DateTime _lastConfigRead = DateTime.MinValue;
    private static readonly string[] _allFunctions = {
        "PI", "E", "GOLDEN_RATIO",
        "Feigenbaum1", "Feigenbaum2", "Feigenbaum3", "Feigenbaum4", "Feigenbaum5",
        "Feigenbaum6", "Feigenbaum7", "Feigenbaum8", "Feigenbaum9", "Feigenbaum10",
        "Feigenbaum11", "Feigenbaum12", "Feigenbaum3D",
        "Apery", "Plastic", "Sqrt2", "Sqrt3", "Sqrt5", "Ln2",
        "SuperGoldenRatio", "Gelfond", "VeryRandom",
        // Все комбинации с PI
        "Apery_div_PI", "PI_div_Apery", "Plastic_div_PI", "PI_div_Plastic",
        "Sqrt2_div_PI", "PI_div_Sqrt2", "Sqrt3_div_PI", "PI_div_Sqrt3",
        "Sqrt5_div_PI", "PI_div_Sqrt5", "Ln2_div_PI", "PI_div_Ln2",
        "SuperGoldenRatio_div_PI", "PI_div_SuperGoldenRatio", "Gelfond_div_PI", "PI_div_Gelfond",
        "GOLDEN_RATIO_div_PI", "PI_div_GOLDEN_RATIO", "E_div_PI", "PI_div_E",
        // Все комбинации с E
        "Apery_div_E", "E_div_Apery", "Plastic_div_E", "E_div_Plastic",
        "Sqrt2_div_E", "E_div_Sqrt2", "Sqrt3_div_E", "E_div_Sqrt3",
        "Sqrt5_div_E", "E_div_Sqrt5", "Ln2_div_E", "E_div_Ln2",
        "SuperGoldenRatio_div_E", "E_div_SuperGoldenRatio", "Gelfond_div_E", "E_div_Gelfond",
        "GOLDEN_RATIO_div_E", "E_div_GOLDEN_RATIO", "PI_div_E", "E_div_PI",
        // Все комбинации с GOLDEN_RATIO
        "Apery_div_GOLDEN", "GOLDEN_div_Apery", "Plastic_div_GOLDEN", "GOLDEN_div_Plastic",
        "Sqrt2_div_GOLDEN", "GOLDEN_div_Sqrt2", "Sqrt3_div_GOLDEN", "GOLDEN_div_Sqrt3",
        "Sqrt5_div_GOLDEN", "GOLDEN_div_Sqrt5", "Ln2_div_GOLDEN", "GOLDEN_div_Ln2",
        "SuperGoldenRatio_div_GOLDEN", "GOLDEN_div_SuperGoldenRatio", "Gelfond_div_GOLDEN", "GOLDEN_div_Gelfond",
        "PI_div_GOLDEN", "GOLDEN_div_PI", "E_div_GOLDEN", "GOLDEN_div_E"
    };

    public static double SelectedFunc
    {
        get
        {
            string funcName = GetSelectedFuncName();
            return GetValueByName(funcName);
        }
    }

    public static double GetValueByName(string funcName)
    {
        switch (funcName)
        {
            // Базовые константы
            case "PI": return PI;
            case "E": return E;
            case "GOLDEN_RATIO": return GOLDEN_RATIO;

            // Константы Фейгенбаума
            case "Feigenbaum1": return Feigenbaum1;
            case "Feigenbaum2": return Feigenbaum2;
            case "Feigenbaum3": return Feigenbaum3;
            case "Feigenbaum4": return Feigenbaum4;
            case "Feigenbaum5": return Feigenbaum5;
            case "Feigenbaum6": return Feigenbaum6;
            case "Feigenbaum7": return Feigenbaum7;
            case "Feigenbaum8": return Feigenbaum8;
            case "Feigenbaum9": return Feigenbaum9;
            case "Feigenbaum10": return Feigenbaum10;
            case "Feigenbaum11": return Feigenbaum11;
            case "Feigenbaum12": return Feigenbaum12;
            case "Feigenbaum3D": return Feigenbaum3D;

            // Дополнительные константы
            case "Apery": return Apery;
            case "Plastic": return Plastic;
            case "Sqrt2": return Sqrt2;
            case "Sqrt3": return Sqrt3;
            case "Sqrt5": return Sqrt5;
            case "Ln2": return Ln2;
            case "SuperGoldenRatio": return SuperGoldenRatio;
            case "Gelfond": return Gelfond;

            // Случайные значения
            case "VeryRandom": return VeryRandom;

            // Комбинации с PI
            case "Apery_div_PI": return Apery_div_PI;
            case "PI_div_Apery": return PI_div_Apery;
            case "Plastic_div_PI": return Plastic_div_PI;
            case "PI_div_Plastic": return PI_div_Plastic;
            case "Sqrt2_div_PI": return Sqrt2_div_PI;
            case "PI_div_Sqrt2": return PI_div_Sqrt2;
            case "Sqrt3_div_PI": return Sqrt3_div_PI;
            case "PI_div_Sqrt3": return PI_div_Sqrt3;
            case "Sqrt5_div_PI": return Sqrt5_div_PI;
            case "PI_div_Sqrt5": return PI_div_Sqrt5;
            case "Ln2_div_PI": return Ln2_div_PI;
            case "PI_div_Ln2": return PI_div_Ln2;
            case "SuperGoldenRatio_div_PI": return SuperGoldenRatio_div_PI;
            case "PI_div_SuperGoldenRatio": return PI_div_SuperGoldenRatio;
            case "Gelfond_div_PI": return Gelfond_div_PI;
            case "PI_div_Gelfond": return PI_div_Gelfond;
            case "GOLDEN_RATIO_div_PI": return GOLDEN_RATIO_div_PI;
            case "PI_div_GOLDEN_RATIO": return PI_div_GOLDEN_RATIO;
            case "PI_div_E": return PI_div_E;

            // Комбинации с E
            case "Apery_div_E": return Apery_div_E;
            case "E_div_Apery": return E_div_Apery;
            case "Plastic_div_E": return Plastic_div_E;
            case "E_div_Plastic": return E_div_Plastic;
            case "Sqrt2_div_E": return Sqrt2_div_E;
            case "E_div_Sqrt2": return E_div_Sqrt2;
            case "Sqrt3_div_E": return Sqrt3_div_E;
            case "E_div_Sqrt3": return E_div_Sqrt3;
            case "Sqrt5_div_E": return Sqrt5_div_E;
            case "E_div_Sqrt5": return E_div_Sqrt5;
            case "Ln2_div_E": return Ln2_div_E;
            case "E_div_Ln2": return E_div_Ln2;
            case "SuperGoldenRatio_div_E": return SuperGoldenRatio_div_E;
            case "E_div_SuperGoldenRatio": return E_div_SuperGoldenRatio;
            case "Gelfond_div_E": return Gelfond_div_E;
            case "E_div_Gelfond": return E_div_Gelfond;
            case "GOLDEN_RATIO_div_E": return GOLDEN_RATIO_div_E;
            case "E_div_GOLDEN_RATIO": return E_div_GOLDEN_RATIO;
            case "E_div_PI": return E_div_PI;

            // Комбинации с GOLDEN_RATIO
            case "Apery_div_GOLDEN": return Apery_div_GOLDEN;
            case "GOLDEN_div_Apery": return GOLDEN_div_Apery;
            case "Plastic_div_GOLDEN": return Plastic_div_GOLDEN;
            case "GOLDEN_div_Plastic": return GOLDEN_div_Plastic;
            case "Sqrt2_div_GOLDEN": return Sqrt2_div_GOLDEN;
            case "GOLDEN_div_Sqrt2": return GOLDEN_div_Sqrt2;
            case "Sqrt3_div_GOLDEN": return Sqrt3_div_GOLDEN;
            case "GOLDEN_div_Sqrt3": return GOLDEN_div_Sqrt3;
            case "Sqrt5_div_GOLDEN": return Sqrt5_div_GOLDEN;
            case "GOLDEN_div_Sqrt5": return GOLDEN_div_Sqrt5;
            case "Ln2_div_GOLDEN": return Ln2_div_GOLDEN;
            case "GOLDEN_div_Ln2": return GOLDEN_div_Ln2;
            case "SuperGoldenRatio_div_GOLDEN": return SuperGoldenRatio_div_GOLDEN;
            case "GOLDEN_div_SuperGoldenRatio": return GOLDEN_div_SuperGoldenRatio;
            case "Gelfond_div_GOLDEN": return Gelfond_div_GOLDEN;
            case "GOLDEN_div_Gelfond": return GOLDEN_div_Gelfond;
            case "PI_div_GOLDEN": return PI_div_GOLDEN;
            case "GOLDEN_div_PI": return GOLDEN_div_PI;
            case "E_div_GOLDEN": return E_div_GOLDEN;
            case "GOLDEN_div_E": return GOLDEN_div_E;

            default: return PI; // Значение по умолчанию
        }
    }

    private static string GetSelectedFuncName()
    {
        lock (_configLock)
        {
            // Проверяем, когда последний раз читали конфиг
            if ((DateTime.Now - _lastConfigRead).TotalSeconds < 5)
                return _cachedFuncName ?? "PI"; // По умолчанию PI

            string configPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "config.ZMath.txt"
            );

            // Значение по умолчанию
            string funcName = "PI";

            try
            {
                if (File.Exists(configPath))
                {
                    funcName = File.ReadAllText(configPath).Trim();

                    // Проверяем, что функция существует
                    if (Array.IndexOf(_allFunctions, funcName) < 0)
                    {
                        funcName = "PI";
                    }

                    _lastConfigRead = DateTime.Now;
                    _cachedFuncName = funcName;
                }
            }
            catch
            {
                // В случае ошибки оставляем предыдущее значение
                return _cachedFuncName ?? "PI";
            }

            return funcName;
        }
    }
}