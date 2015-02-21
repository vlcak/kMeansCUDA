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
        //private String fileName1, fileName2;
        private Point[] points1 = null, points2 = null;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void buttonLoadFiles1_Click(object sender, RoutedEventArgs e)
        {
            var ofd = new OpenFileDialog { DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat" };

            bool? result = ofd.ShowDialog();


            if (result == true) ;
            {
                points1 = loadData(ofd.FileName);
                labelFileName1.Content = ofd.FileName;
                if (points2 != null)
                {
                    ComparePoints(points1, points2);
                }
            }
        }

        private void buttonLoadFiles2_Click(object sender, RoutedEventArgs e)
        {
            var ofd = new OpenFileDialog { DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat" };

            bool? result = ofd.ShowDialog();


            if (result == true) ;
            {
                points2 = loadData(ofd.FileName);
                labelFileName2.Content = ofd.FileName;
                if (points1 != null)
                {
                    ComparePoints(points1, points2);
                }
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
                            //p.DistanceFromCluster = br.ReadSingle();
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
            listBoxDifferentFiles.Items.Clear();
            int differencesCount = 0;
            if (a.Length != b.Length)
            {
                listBoxDifferentFiles.Items.Add("Files cointains different number of points (" + a.Length + " : " + b.Length + ")");
                ++differencesCount;
            }
            else
            {
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i].Cluster != b[i].Cluster)
                    {
                        listBoxDifferentFiles.Items.Add("Points " + i + " are assigned to different cluster (" + a[i].Cluster + " : " + b[i].Cluster + ") distances from cluster: " + a[i].DistanceFromCluster + " " + b[i].DistanceFromCluster);
                        ++differencesCount;
                    }
                    else
                    {
                        for (int j = 0; j < a[i].Coords.Length; j++)
                        {
                            if (Math.Abs(a[i].Coords[j] - b[i].Coords[j]) > 0.0001)
                            {
                                listBoxDifferentFiles.Items.Add("Points " + i + " have different coordinate (coordinate: " + j + ", values: " + a[i].Coords[j] + " : " + b[i].Coords[j] + ")");
                                ++differencesCount;
                            }
                        }
                    }
                }
            }
            listBoxDifferentFiles.Items.Add("Compare complete, differences: " + differencesCount);
        }

        private void ButtonRefresh_OnClick(object sender, RoutedEventArgs e)
        {
            points1 = loadData(labelFileName1.Content.ToString());
            points2 = loadData(labelFileName2.Content.ToString());
            ComparePoints(points1, points2);
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
        public float DistanceFromCluster;
    }
}
