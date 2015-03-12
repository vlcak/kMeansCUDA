using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Visualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private readonly Color[] _myColors;

        public MainWindow()
        {
            _myColors = new[]
            {
                Colors.Red,
                Colors.Green,
                Colors.Blue,
                Colors.Pink,
                Colors.DeepPink,
                Colors.Lime,
                Colors.MediumSpringGreen,
                Colors.DarkOliveGreen,
                Colors.Aqua,
                Colors.MediumTurquoise,
                Colors.Purple,
                Colors.MediumPurple,
                Colors.Yellow,
                Colors.Orange,
                Colors.OrangeRed,
                Colors.White,
                Colors.Black,
                Colors.Gray,
                Colors.DarkGray,
                Colors.Chocolate,
                Colors.Coral,
                Colors.Crimson,
                Colors.DarkMagenta,
                Colors.DarkSeaGreen,
                Colors.DodgerBlue,
                Colors.Fuchsia,
                Colors.Gold,
                Colors.Goldenrod,
                Colors.GreenYellow, 
                Colors.Indigo,
                Colors.OliveDrab,
                Colors.Teal
            };
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var cofd = new Microsoft.Win32.OpenFileDialog {DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat"};

            bool? result1 = cofd.ShowDialog();

            var mofd = new Microsoft.Win32.OpenFileDialog {DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat"};

            bool? result2 = mofd.ShowDialog();

            if (result1 == true && result2 == true)
            {
                Point[] points = loadData(cofd.FileName);
                PrintPoints(points, false);
                points = loadData(mofd.FileName);
                PrintPoints(points, true);
            }
        }

        private Point[] loadData(string fileName)
        {
            var points = new LinkedList<Point>();
            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                using (var br = new BinaryReader(fs))
                {
                    var dimensions = br.ReadUInt64();
                    while (true)
                    {
                        try
                        {
                            var p = new Point(dimensions);
                            for (ulong i = 0; i < dimensions; i++)
                            {
                                p.Coords[i] = br.ReadSingle();
                            }
                            p.Cluster = br.ReadByte();
                            points.AddLast(p);
                        }
                        catch (EndOfStreamException)
                        {
                            break;
                        }
                    }
                }
            }
            return points.ToArray();
        }

        private void PrintPoints(Point[] points, bool mean)
        {
            //OrthographicCamera orthographicCamera = new OrthographicCamera();
            //orthographicCamera.Position = new Point3D(0,0,2);
            //orthographicCamera.LookDirection = new Vector3D(0,0,-1);

            for (int i = 0; i < points.Length; i++)
            {
                Shape s;
                if (mean)
                {
                    s = new Ellipse {Stroke = new SolidColorBrush(Colors.Black), Width = 10, Height = 10};
                }
                else
                {
                    s = new Rectangle {Width = 5, Height = 5};
                }
                s.Fill = new SolidColorBrush(_myColors[mean ? i : points[i].Cluster]);
                s.MouseEnter += r_MouseEnter;
                s.MouseLeave += delegate { ClusterName.Content = "Cluster:"; };
                Canvas.SetLeft(s, Graph.ActualWidth / 4 + points[i].Coords[0] / 20f * Graph.ActualWidth);
                Canvas.SetTop(s, Graph.ActualHeight / 4 + points[i].Coords[1] / 20f * Graph.ActualHeight);
                Graph.Children.Add(s);
            }

 
        }

        void r_MouseEnter(object sender, MouseEventArgs e)
        {
            var shape = sender as Shape;
            if (shape != null)
            {
                ClusterName.Content = "Cluster: " + FindCluster(((SolidColorBrush)shape.Fill).Color);
            }
        }

        private int FindCluster(Color c)
        {
            for (int i = 0; i < _myColors.Length; i++)
            {
                if (_myColors[i] == c)
                {
                    return i;
                }
            }
            return 0;
        }

        /*private float getMinMax(Point[] points)
        {
            float[] value = new float[4];
            for (int i = 0; i < points.Length; i++)
            {
                if (poin)
            }
        }*/
    }

    internal class Point
    {
        public Point(ulong dimensions)
        {
            Coords = new float[dimensions];
        }

        public int Cluster;
        public float[] Coords;
    }


}
