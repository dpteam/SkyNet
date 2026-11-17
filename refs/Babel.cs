using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace BabelFTP
{
    // ======================================================================
    // КЛАСС ДЛЯ ЛОГИРОВАНИЯ (Trace + Цветная консоль + Файл)
    // ======================================================================
    public class ColorConsoleTraceListener : TraceListener
    {
        public override void Write(string message) { Write(message, ""); }
        public override void WriteLine(string message) { WriteLine(message, ""); }

        public override void WriteLine(string message, string category)
        {
            ConsoleColor originalColor = Console.ForegroundColor;
            switch (category)
            {
                case "INFO": Console.ForegroundColor = ConsoleColor.Green; break;
                case "WARN": Console.ForegroundColor = ConsoleColor.Yellow; break;
                case "ERROR": Console.ForegroundColor = ConsoleColor.Red; break;
                case "FTP": Console.ForegroundColor = ConsoleColor.Cyan; break;
                case "TUI": Console.ForegroundColor = ConsoleColor.Magenta; break;
                default: Console.ForegroundColor = ConsoleColor.Gray; break;
            }
            Console.WriteLine("[" + DateTime.Now.ToString("HH:mm:ss") + "] [" + category + "] " + message);
            Console.ForegroundColor = originalColor;
        }

        public override void Write(string message, string category)
        {
            WriteLine(message, category);
        }
    }

    public class FileTraceListener : TraceListener
    {
        private StreamWriter writer;

        public FileTraceListener(string filename)
        {
            writer = new StreamWriter(filename, true) { AutoFlush = true };
        }

        public override void Write(string message) { Write(message, ""); }
        public override void WriteLine(string message) { WriteLine(message, ""); }

        public override void WriteLine(string message, string category)
        {
            writer.WriteLine("[" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "] [" + category + "] " + message);
        }

        public override void Write(string message, string category)
        {
            WriteLine(message, category);
        }

        public override void Close()
        {
            writer.Close();
            base.Close();
        }
    }

    // ======================================================================
    // TUI ХЕЛПЕРЫ (Прогресс бары, меню, статусы)
    // ======================================================================
    public static class TUI
    {
        public static void SetStatus(string status)
        {
            Console.Title = "BabelFTP Server - " + status;
        }

        public static void DrawProgressBar(int progress, int total)
        {
            Console.Write("\r[");
            int width = 30;
            int filled = (int)((double)progress / total * width);
            for (int i = 0; i < width; i++)
            {
                if (i < filled) Console.Write("#");
                else Console.Write("-");
            }
            Console.Write("] " + progress + "/" + total + "   ");
        }

        public static int ShowMenu(string title, string[] options)
        {
            int selected = 0;
            ConsoleKey key;
            do
            {
                Console.Clear();
                Trace.WriteLine(title, "TUI");
                for (int i = 0; i < options.Length; i++)
                {
                    if (i == selected)
                    {
                        Console.BackgroundColor = ConsoleColor.Blue;
                        Console.ForegroundColor = ConsoleColor.White;
                    }
                    Console.WriteLine("  " + options[i]);
                    Console.ResetColor();
                }
                key = Console.ReadKey(true).Key;

                if (key == ConsoleKey.UpArrow) selected--;
                if (key == ConsoleKey.DownArrow) selected++;
                if (selected < 0) selected = options.Length - 1;
                if (selected >= options.Length) selected = 0;

            } while (key != ConsoleKey.Enter);

            return selected;
        }
    }

    // ======================================================================
    // ВИРТУАЛЬНАЯ ФАЙЛОВАЯ СИСТЕМА (Вавилонская библиотека)
    // ======================================================================
    public class BabelFS
    {
        // Алфавит Вавилонской библиотеки
        private static readonly char[] alphabet = "abcdefghijklmnopqrstuvwxyz".ToCharArray();
        private const int MaxDepth = 5; // Ограничение глубины рекурсии для листинга, чтобы не повесить клиент

        // Проверка существования пути (всегда true для директорий a-z, файл - babel.txt)
        public bool DirectoryExists(string path)
        {
            return true; // Виртуально существует всё
        }

        public bool FileExists(string path)
        {
            return path.EndsWith("babel.txt", StringComparison.OrdinalIgnoreCase);
        }

        public string[] GetDirectories(string path)
        {
            List<string> dirs = new List<string>();
            foreach (char c in alphabet)
            {
                dirs.Add(c.ToString());
            }
            return dirs.ToArray();
        }

        public string[] GetFiles(string path)
        {
            return new string[] { "babel.txt" };
        }

        // Генерация контента файла на основе пути
        public byte[] GetFileContent(string path)
        {
            // Простая детерминированная генерация текста (410 символов как в оригинальной книге)
            int seed = 0;
            foreach (char c in path)
            {
                seed = (seed * 31) + c;
            }
            Random rng = new Random(seed);
            char[] chars = new char[410];
            for (int i = 0; i < 410; i++)
            {
                chars[i] = (char)rng.Next(97, 123); // a-z
            }
            return Encoding.ASCII.GetBytes(new string(chars) + "\r\n");
        }
    }

    // ======================================================================
    // FTP СЕРВЕР И КЛИЕНТ
    // ======================================================================
    public class FtpServer
    {
        private TcpListener listener;
        private int port;
        private bool isRunning = false;
        private BabelFS fs = new BabelFS();
        private int activeClients = 0;

        public string MasqueradeIp { get; set; }
        public int PasvPortMin { get; set; }
        public int PasvPortMax { get; set; }
        private int currentPasvPort = 0;

        public FtpServer(int port)
        {
            this.port = port;
        }

        // Потокобезопасное получение следующего порта из диапазона
        private int GetNextPasvPort()
        {
            lock (this)
            {
                if (PasvPortMin > 0 && PasvPortMax >= PasvPortMin)
                {
                    int p = currentPasvPort;
                    if (p < PasvPortMin || p > PasvPortMax) p = PasvPortMin;
                    currentPasvPort = p + 1;
                    if (currentPasvPort > PasvPortMax) currentPasvPort = PasvPortMin;
                    return p;
                }
                return 0; // 0 = ОС выберет случайный порт
            }
        }

        public void Start()
        {
            listener = new TcpListener(IPAddress.Any, port);
            listener.Start();
            isRunning = true;
            Trace.WriteLine("FTP Server started on port " + port, "INFO");
            TUI.SetStatus("Running (Port " + port + ")");

            while (isRunning)
            {
                try
                {
                    TcpClient client = listener.AcceptTcpClient();
                    Thread clientThread = new Thread(new ParameterizedThreadStart(HandleClient));
                    clientThread.IsBackground = true;
                    clientThread.Start(client);
                }
                catch (SocketException)
                {
                    // Игнорируем, сервер останавливается
                }
            }
        }

        public void Stop()
        {
            isRunning = false;
            listener.Stop();
        }

        private void HandleClient(object obj)
        {
            TcpClient client = (TcpClient)obj;
            Interlocked.Increment(ref activeClients);
            TUI.SetStatus("Running - " + activeClients + " active client(s)");

            string clientIP = ((IPEndPoint)client.Client.RemoteEndPoint).Address.ToString();
            Trace.WriteLine("Client connected: " + clientIP, "FTP");

            NetworkStream stream = null;
            StreamReader reader = null;
            StreamWriter writer = null;

            try
            {
                stream = client.GetStream();
                reader = new StreamReader(stream, Encoding.ASCII);
                writer = new StreamWriter(stream, Encoding.ASCII) { AutoFlush = true };

                writer.WriteLine("220 BabelFTP Service Ready. Read-only access.");

                string currentDir = "/";
                string dataType = "A"; // ASCII по умолчанию
                IPEndPoint dataEndPoint = null;
                TcpListener passiveListener = null;

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (line.Length == 0) continue;

                    string[] parts = line.Split(new char[] { ' ' }, 2);
                    string cmd = parts[0].ToUpper();
                    string args = parts.Length > 1 ? parts[1].Trim() : "";

                    // Добавляем IP-адрес клиента прямо в строку лога команды
                    Trace.WriteLine("[" + clientIP + "] CMD: " + cmd + (args.Length > 0 ? " " + args : ""), "FTP");

                    switch (cmd)
                    {
                        case "AUTH":
                            // Отклоняем запрос на TLS/SSL, так как сервер работает только в Plain Text
                            writer.WriteLine("502 Security mechanism not implemented.");
                            break;
                        case "USER":
                        case "PASS":
                            writer.WriteLine("230 User logged in.");
                            break;
                        case "SYST":
                            writer.WriteLine("215 UNIX Type: L8");
                            break;
                        case "FEAT":
                            writer.WriteLine("211-Features:");
                            writer.WriteLine(" SIZE");
                            writer.WriteLine("211 End");
                            break;
                        case "PWD":
                            writer.WriteLine("257 \"" + currentDir + "\" is current directory.");
                            break;
                        case "TYPE":
                            dataType = args.Length > 0 ? args.Substring(0, 1).ToUpper() : "A";
                            writer.WriteLine("200 Type set to " + dataType + ".");
                            break;
                        case "CWD":
                            string newDir = NormalizePath(currentDir, args);
                            // В нашей Вавилонской библиотеке любая директория из алфавита существует
                            currentDir = newDir;
                            writer.WriteLine("250 Directory changed to " + currentDir);
                            break;
                        case "CDUP":
                            if (currentDir != "/")
                            {
                                currentDir = currentDir.TrimEnd('/');
                                int idx = currentDir.LastIndexOf('/');
                                currentDir = idx >= 0 ? currentDir.Substring(idx) : "/";
                                if (currentDir == "") currentDir = "/";
                            }
                            writer.WriteLine("250 Directory changed to " + currentDir);
                            break;
                        case "PASV":
                            int pasvPort = GetNextPasvPort();
                            try
                            {
                                passiveListener = new TcpListener(IPAddress.Any, pasvPort);
                                passiveListener.Start();
                            }
                            catch (SocketException)
                            {
                                // Если порт занят, просим ОС дать любой свободный
                                passiveListener = new TcpListener(IPAddress.Any, 0);
                                passiveListener.Start();
                            }

                            byte[] ipBytes;
                            if (!string.IsNullOrEmpty(MasqueradeIp))
                                ipBytes = IPAddress.Parse(MasqueradeIp).GetAddressBytes();
                            else
                            {
                                IPEndPoint clientLocalEp = (IPEndPoint)client.Client.LocalEndPoint;
                                ipBytes = clientLocalEp.Address.GetAddressBytes();
                            }

                            if (ipBytes.Length != 4) ipBytes = new byte[] { 127, 0, 0, 1 };

                            IPEndPoint passiveEp = (IPEndPoint)passiveListener.LocalEndpoint;
                            int p1 = passiveEp.Port / 256;
                            int p2 = passiveEp.Port % 256;

                            writer.WriteLine(string.Format("227 Entering Passive Mode ({0},{1},{2},{3},{4},{5})",
                                ipBytes[0], ipBytes[1], ipBytes[2], ipBytes[3], p1, p2));
                            break;

                        case "EPSV": // Расширенный пассивный режим (предпочтителен для FileZilla)
                            try
                            {
                                passiveListener = new TcpListener(IPAddress.Any, GetNextPasvPort());
                                passiveListener.Start();
                            }
                            catch (SocketException)
                            {
                                passiveListener = new TcpListener(IPAddress.Any, 0);
                                passiveListener.Start();
                            }
                            IPEndPoint epsvEp = (IPEndPoint)passiveListener.LocalEndpoint;
                            // EPSV не требует передачи IP-адреса, клиент использует тот же, что и для управления
                            writer.WriteLine("229 Entering Extended Passive Mode (|||" + epsvEp.Port + "|)");
                            break;

                        case "LIST":
                        case "NLST":
                            // ВАЖНО: Отправляем 150 ДО того, как ждем подключение клиента!
                            writer.WriteLine("150 Opening data connection for " + cmd + ".");

                            TcpClient dataClient = GetDataConnection(ref passiveListener, ref dataEndPoint, clientIP);
                            if (dataClient != null)
                            {
                                using (NetworkStream dataStream = dataClient.GetStream())
                                using (StreamWriter dataWriter = new StreamWriter(dataStream, Encoding.ASCII) { AutoFlush = true })
                                {
                                    string[] dirs = fs.GetDirectories(currentDir);
                                    string[] files = fs.GetFiles(currentDir);

                                    foreach (string d in dirs)
                                    {
                                        if (cmd == "LIST")
                                            dataWriter.WriteLine(string.Format("drwxr-xr-x 1 ftp ftp 0 Jan 01 00:00 {0}", d));
                                        else
                                            dataWriter.WriteLine(d);
                                    }
                                    foreach (string f in files)
                                    {
                                        byte[] content = fs.GetFileContent(currentDir + "/" + f);
                                        if (cmd == "LIST")
                                            dataWriter.WriteLine(string.Format("-rw-r--r-- 1 ftp ftp {0} Jan 01 00:00 {1}", content.Length, f));
                                        else
                                            dataWriter.WriteLine(f);
                                    }
                                }
                                dataClient.Close();
                                writer.WriteLine("226 Transfer complete.");
                            }
                            else
                            {
                                writer.WriteLine("425 Cannot open data connection. Check router port forwarding for passive ports.");
                            }

                            // Очищаем слушателя после использования
                            if (passiveListener != null) { passiveListener.Stop(); passiveListener = null; }
                            dataEndPoint = null;
                            break;
                        case "PORT":
                            try
                            {
                                string[] p = args.Split(',');
                                byte[] ip = new byte[4] { byte.Parse(p[0]), byte.Parse(p[1]), byte.Parse(p[2]), byte.Parse(p[3]) };
                                int port = int.Parse(p[4]) * 256 + int.Parse(p[5]);
                                dataEndPoint = new IPEndPoint(new IPAddress(ip), port);
                                writer.WriteLine("200 PORT command successful.");
                            }
                            catch
                            {
                                writer.WriteLine("501 Syntax error in parameters.");
                            }
                            break;
                        case "RETR":
                            string filePath = NormalizePath(currentDir, args);
                            if (fs.FileExists(filePath))
                            {
                                TcpClient dlClient = GetDataConnection(ref passiveListener, ref dataEndPoint, clientIP);
                                if (dlClient != null)
                                {
                                    writer.WriteLine("150 Opening data connection for RETR.");
                                    byte[] content = fs.GetFileContent(filePath);
                                    using (NetworkStream dataStream = dlClient.GetStream())
                                    {
                                        dataStream.Write(content, 0, content.Length);
                                    }
                                    dlClient.Close();
                                    writer.WriteLine("226 Transfer complete.");
                                }
                                else
                                {
                                    writer.WriteLine("425 Cannot open data connection.");
                                }
                            }
                            else
                            {
                                writer.WriteLine("550 File not found or access denied.");
                            }
                            break;
                        case "SIZE":
                            string szPath = NormalizePath(currentDir, args);
                            if (fs.FileExists(szPath))
                            {
                                byte[] szContent = fs.GetFileContent(szPath);
                                writer.WriteLine("213 " + szContent.Length);
                            }
                            else
                            {
                                writer.WriteLine("550 File not found.");
                            }
                            break;
                        case "QUIT":
                            writer.WriteLine("221 Goodbye.");
                            break;
                        // Read-only violations
                        case "STOR":
                        case "MKD":
                        case "RMD":
                        case "DELE":
                        case "RNFR":
                        case "RNTO":
                            writer.WriteLine("550 Permission denied. BabelFTP is strictly READ-ONLY.");
                            break;
                        default:
                            writer.WriteLine("502 Command not implemented.");
                            break;
                    }
                }
            }
            catch (IOException ioEx)
            {
                Trace.WriteLine("Client disconnected unexpectedly: " + ioEx.Message, "WARN");
            }
            catch (Exception ex)
            {
                Trace.WriteLine("Error handling client: " + ex.Message, "ERROR");
            }
            finally
            {
                if (client != null) client.Close();
                Interlocked.Decrement(ref activeClients);
                if (activeClients < 0) activeClients = 0;
                TUI.SetStatus("Running - " + activeClients + " active client(s)");
                Trace.WriteLine("Client " + clientIP + " disconnected.", "FTP");
            }
        }

        private TcpClient GetDataConnection(ref TcpListener passiveListener, ref IPEndPoint dataEndPoint, string clientIP)
        {
            try
            {
                if (passiveListener != null)
                {
                    IAsyncResult ar = passiveListener.BeginAcceptTcpClient(null, null);
                    if (ar.AsyncWaitHandle.WaitOne(20000, false))
                    {
                        return passiveListener.EndAcceptTcpClient(ar);
                    }
                    Trace.WriteLine("[" + clientIP + "] Пассивное соединение отклонено (Таймаут 20с). Проверьте проброс портов.", "WARN");
                    return null;
                }
                else if (dataEndPoint != null)
                {
                    TcpClient dc = new TcpClient();
                    IAsyncResult ar = dc.BeginConnect(dataEndPoint.Address, dataEndPoint.Port, null, null);
                    if (ar.AsyncWaitHandle.WaitOne(20000, false))
                    {
                        dc.EndConnect(ar);
                        return dc;
                    }
                    Trace.WriteLine("[" + clientIP + "] Активное (PORT) соединение не удалось (Таймаут 20с).", "WARN");
                    dc.Close();
                    return null;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine("[" + clientIP + "] Data connection failed: " + ex.Message, "ERROR");
            }
            return null;
        }

        private string NormalizePath(string current, string target)
        {
            if (target.StartsWith("/")) return target;
            if (target == "..")
            {
                if (current == "/") return "/";
                string temp = current.TrimEnd('/');
                int idx = temp.LastIndexOf('/');
                return idx >= 0 ? temp.Substring(0, idx + 1) : "/";
            }
            if (current.EndsWith("/")) return current + target;
            return current + "/" + target;
        }
    }

    // ======================================================================
    // ГЛАВНЫЙ КЛАСС ПРОГРАММЫ
    // ======================================================================
    public class Program
    {
        static Dictionary<string, string> config = new Dictionary<string, string>();

        static void Main(string[] args)
        {
            SetupLogging();
            TUI.SetStatus("Initializing...");

            // Автосоздание конфига
            string configFile = "babelftp.conf";
            if (args.Length > 0)
            {
                foreach (string arg in args)
                {
                    if (arg.StartsWith("--config="))
                    {
                        configFile = arg.Substring(9);
                    }
                }
            }

            if (!File.Exists(configFile))
            {
                File.WriteAllText(configFile, "# BabelFTP Configuration\r\nport=21\r\nmaxdepth=5\r\nmasquerade_ip=\r\n# Диапазон пассивных портов (ОБЯЗАТЕЛЬНО пробросьте их в роутере!)\r\npasv_port_min=50000\r\npasv_port_max=50050\r\n");
                Trace.WriteLine("Configuration file created: " + configFile, "INFO");
            }

            LoadConfig(configFile);

            bool isInteractive = true;
            int port = 21;
            string masqueradeIp = config.ContainsKey("masquerade_ip") ? config["masquerade_ip"] : "";
            int pasvMin = config.ContainsKey("pasv_port_min") ? int.Parse(config["pasv_port_min"]) : 50000;
            int pasvMax = config.ContainsKey("pasv_port_max") ? int.Parse(config["pasv_port_max"]) : 50050;

            foreach (string arg in args)
            {
                if (arg.StartsWith("--port=")) { int.TryParse(arg.Substring(7), out port); isInteractive = false; }
                else if (arg.StartsWith("--masquerade=")) { masqueradeIp = arg.Substring(13); isInteractive = false; }
                else if (arg.StartsWith("--pasv_min=")) { int.TryParse(arg.Substring(11), out pasvMin); isInteractive = false; }
                else if (arg.StartsWith("--pasv_max=")) { int.TryParse(arg.Substring(11), out pasvMax); isInteractive = false; }
                else if (arg == "--help" || arg == "-h") { PrintHelp(); return; }
            }

            foreach (string arg in args)
            {
                if (arg.StartsWith("--port="))
                {
                    int.TryParse(arg.Substring(7), out port);
                    isInteractive = false;
                }
                else if (arg.StartsWith("--masquerade="))
                {
                    masqueradeIp = arg.Substring(13);
                    isInteractive = false;
                }
                else if (arg == "--help" || arg == "-h")
                {
                    PrintHelp();
                    return;
                }
            }

            // Если запускают интерактивно, спрашиваем IP
            if (isInteractive)
            {
                Trace.WriteLine("Interactive mode started. Use --help for CLI usage.", "TUI");
                int choice = TUI.ShowMenu("BabelFTP Setup", new string[] {
                    "Start on default port (21)",
                    "Start on custom port (8021)",
                    "Start on random port",
                    "Exit"
                });

                switch (choice)
                {
                    case 0: port = 21; break;
                    case 1: port = 8021; break;
                    case 2: port = new Random().Next(1000, 9999); break;
                    case 3: return;
                }

                // Спрашиваем про внешний IP для NAT
                Console.WriteLine("\nEnter public IP for NAT/Hairpin support (leave empty for local only):");
                Console.Write("> ");
                masqueradeIp = Console.ReadLine().Trim();
            }

            // Динамическая отказоустойчивость при запуске сервера
            try
            {
                FtpServer server = new FtpServer(port);
                server.MasqueradeIp = masqueradeIp;
                server.PasvPortMin = pasvMin;
                server.PasvPortMax = pasvMax;

                // Симуляция прогресса
                Trace.WriteLine("Starting virtual filesystem...", "INFO");
                for (int i = 0; i <= 100; i += 20)
                {
                    TUI.DrawProgressBar(i, 100);
                    Thread.Sleep(100);
                }
                Console.WriteLine();

                Trace.WriteLine("Masquerade IP set to: " + (string.IsNullOrEmpty(masqueradeIp) ? "Auto (Local)" : masqueradeIp), "INFO");
                server.Start();
            }
            catch (SocketException ex)
            {
                if (ex.ErrorCode == 10048) // WSAEADDRINUSE
                {
                    Trace.WriteLine("Port " + port + " is already in use. Please choose another port.", "ERROR");
                }
                else
                {
                    Trace.WriteLine("Socket Error: " + ex.Message, "ERROR");
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine("Fatal Error: " + ex.Message, "ERROR");
            }

            Trace.WriteLine("Press Enter to exit...", "INFO");
            Console.ReadLine();
        }

        static void PrintHelp()
        {
            Console.WriteLine("BabelFTP - Virtual Library of Babel FTP Server");
            Console.WriteLine("Usage: BabelFTP.exe [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --help, -h          Show this help message.");
            Console.WriteLine("  --port=<port>       Specify the port to listen on (e.g., --port=21).");
            Console.WriteLine("  --masquerade=<ip>   Specify public IP for NAT/Hairpin support (e.g., --masquerade=203.0.113.5).");
            Console.WriteLine("  --config=<file>     Specify a configuration file (default: babelftp.conf).");
            Console.WriteLine("");
            Console.WriteLine("Supported FTP Commands: STOR(denied), LIST, TYPE, PORT, PASV, USER, PASS, RETR, CWD, PWD, QUIT, MKD(denied), RMD(denied), CDUP, RNFR(denied), RNTO(denied), DELE(denied)");
        }

        static void SetupLogging()
        {
            // Добавляем цветной консольный слушатель
            Trace.Listeners.Add(new ColorConsoleTraceListener());

            // Добавляем файловый слушатель с новым дататаймом
            string logFileName = "babelftp_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".log";
            Trace.Listeners.Add(new FileTraceListener(logFileName));

            // Убираем Default слушатель, чтобы не дублировать вывод в Output
            Trace.Listeners.Remove("Default");
        }

        static void LoadConfig(string file)
        {
            try
            {
                string[] lines = File.ReadAllLines(file);
                foreach (string line in lines)
                {
                    if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;
                    string[] parts = line.Split(new char[] { '=' }, 2);
                    if (parts.Length == 2)
                    {
                        config[parts[0].Trim().ToLower()] = parts[1].Trim();
                    }
                }
                Trace.WriteLine("Configuration loaded successfully.", "INFO");
            }
            catch (Exception ex)
            {
                Trace.WriteLine("Failed to load config: " + ex.Message, "WARN");
            }
        }
    }
}