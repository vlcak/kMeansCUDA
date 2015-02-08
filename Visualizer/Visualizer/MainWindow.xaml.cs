using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Visualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private float lowerBound = 0f, upperBound = 10f;

        private Color[] myColors;

        public MainWindow()
        {

            myColors = new Color[]
            {
                Colors.Pink,
                Colors.DeepPink,
                Colors.Red,
                Colors.Green,
                Colors.Lime,
                Colors.MediumSpringGreen,
                Colors.DarkOliveGreen,
                Colors.Blue,
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
            Microsoft.Win32.OpenFileDialog cofd = new Microsoft.Win32.OpenFileDialog();
            
            cofd.DefaultExt = ".dat";
            cofd.Filter = "Data files (*.dat)|*.dat";
            bool? result1 = cofd.ShowDialog();

            Microsoft.Win32.OpenFileDialog mofd = new Microsoft.Win32.OpenFileDialog();
            
            mofd.DefaultExt = ".dat";
            mofd.Filter = "Data files (*.dat)|*.dat";
            bool? result2 = mofd.ShowDialog();

            if (result1 == true && result2 == true)
            {
                Point[] points = loadData(cofd.FileName);
                printPoints(points, false);
                points = loadData(mofd.FileName);
                printPoints(points, true);
            }
        }

        private Point[] loadData(string fileName)
        {
            ulong dimensions;
            LinkedList<Point> points = new LinkedList<Point>();
            using (FileStream fs = new FileStream(fileName, FileMode.Open))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    dimensions = br.ReadUInt64();
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

        private void printPoints(Point[] points, bool mean)
        {
            //OrthographicCamera orthographicCamera = new OrthographicCamera();
            //orthographicCamera.Position = new Point3D(0,0,2);
            //orthographicCamera.LookDirection = new Vector3D(0,0,-1);

            for (int i = 0; i < points.Length; i++)
            {
                Shape s;
                if (mean)
                {
                    s = new Ellipse();
                    s.Stroke = new SolidColorBrush(Colors.Black);
                    s.Width = 10;
                    s.Height = 10;
                }
                else
                {
                    s = new Rectangle();
                    s.Width = 5;
                    s.Height = 5;
                }
                s.Fill = new SolidColorBrush(myColors[mean ? i : points[i].Cluster]);
                s.MouseEnter += r_MouseEnter;
                s.MouseLeave += delegate(object sender, MouseEventArgs args) { ClusterName.Content = "Cluster:"; };
                Canvas.SetLeft(s, Graph.ActualWidth / 4 + points[i].Coords[0] / 20f * Graph.ActualWidth);
                Canvas.SetTop(s, Graph.ActualHeight / 4 + points[i].Coords[1] / 20f * Graph.ActualHeight);
                Graph.Children.Add(s);
            }

 
        }

        void r_MouseEnter(object sender, MouseEventArgs e)
        {
            if (sender is Shape)
            {
                ClusterName.Content = "Cluster: " + findCluster(((SolidColorBrush)((Shape)sender).Fill).Color);
            }
        }

        private int findCluster(Color c)
        {
            for (int i = 0; i < myColors.Length; i++)
            {
                if (myColors[i] == c)
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
