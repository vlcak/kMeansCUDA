using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using Microsoft.Win32;

namespace Comparer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void buttonLoadFiles_Click(object sender, RoutedEventArgs e)
        {
            var ofd1 = new OpenFileDialog { DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat" };

            bool? result1 = ofd1.ShowDialog();

            var ofd2 = new OpenFileDialog { DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat" };

            bool? result2 = ofd2.ShowDialog();

            if (result1 == true && result2 == true)
            {
                Point[] points1 = loadData(ofd1.FileName);
                Point[] points2 = loadData(ofd2.FileName);
                ComparePoints(points1, points2);
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

        private void ComparePoints(Point[] a, Point[] b)
        {
            if (a.Length != b.Length)
            {
                listBoxDifferentFiles.Items.Add("Files cointains different number of points (" + a.Length + " : " + b.Length + ")");
            }
            else
            {
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i].Cluster != b[i].Cluster)
                    {
                        listBoxDifferentFiles.Items.Add("Points " + i + " are assigned to different cluster (" + a[i].Cluster+ " : " + b[i].Cluster + ")");
                    }
                    else
                    {
                        for (int j = 0; j < a[i].Coords.Length; j++)
                        {
                            if (Math.Abs(a[i].Coords[j] - b[i].Coords[j]) > 0.0001)
                            {
                                listBoxDifferentFiles.Items.Add("Points " + i + " have different coordinate (coordinate: " + j + ", values: " + a[i].Coords[j] + " : " + b[i].Coords[j] + ")");
                            }
                        }
                    }
                }
            }
            listBoxDifferentFiles.Items.Add("Compare complete");
        }
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
