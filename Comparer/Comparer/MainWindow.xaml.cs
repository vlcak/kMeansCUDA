using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
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
        private String path;
        public MainWindow()
        {
            path = "..\\..\\..\\..\\";

            ProjectData[] projectData = new[]
            {
                new ProjectData {Label = "Serial", Path = "k-means-serial\\k-means-serial\\"},
                new ProjectData {Label = "TBB", Path = "k-meansTBB\\k-means\\"},
                new ProjectData {Label = "TBB_SSE", Path = "k-meansTBB_SSE\\k-means\\"},
                new ProjectData {Label = "CUDA", Path = "k-meansCUDA\\k-meansCUDA\\"}
            };

            InitializeComponent();
            foreach (var project in projectData)
            {
                ComboBoxLocation1.Items.Add(project);
                ComboBoxLocation2.Items.Add(project);
            }
            ComboBoxDimension.Items.Add("2");
            ComboBoxDimension.Items.Add("3");
            ComboBoxDimension.Items.Add("4");
            ComboBoxDimension.Items.Add("8");
            ComboBoxDimension.SelectedItem = "3";
        }

        private void buttonLoadFiles1_Click(object sender, RoutedEventArgs e)
        {
            var ofd = new OpenFileDialog { DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat" };

            bool? result = ofd.ShowDialog();


            if (result != null && result.Value)
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


            if (result != null && result.Value)
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
                            if (Double.IsNaN(a[i].Coords[j]) || Double.IsNaN(b[i].Coords[j]) || (Math.Abs(a[i].Coords[j] - b[i].Coords[j]) > 0.0001))
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

        private void ButtonSwitch_OnClick(object sender, RoutedEventArgs e)
        {
            string file1 = labelFileName1.Content.ToString();
            string file2 = labelFileName2.Content.ToString();
            if (!file1.Contains("clusters"))
            {
                file1 = file1.Insert(file1.LastIndexOf("means"), "clusters");
                file1 = file1.Remove(file1.LastIndexOf("means"), 5);
                file2 = file2.Insert(file2.LastIndexOf("means"), "clusters");
                file2 = file2.Remove(file2.LastIndexOf("means"), 5);
            }
            else
            {
                file1 = file1.Replace("clusters", "means");
                file2 = file2.Replace("clusters", "means");
            }

            points1 = loadData(file1);
            points2 = loadData(file2);
            labelFileName1.Content = file1;
            labelFileName2.Content = file2;
            ComparePoints(points1, points2);
        }

        private void ComboBoxLocation1_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            string filePath1 = string.Format("{0}{1}clusters{2}D.dat", path, ((ProjectData) e.AddedItems[0]).Path, ComboBoxDimension.SelectedItem);
            points1 = loadData(filePath1);
            labelFileName1.Content = filePath1;
            if (points2 != null)
            {
                ComparePoints(points1, points2);
            }
        }

        private void ComboBoxLocation2_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            string filePath2 = string.Format("{0}{1}clusters{2}D.dat", path, ((ProjectData)e.AddedItems[0]).Path, ComboBoxDimension.SelectedItem);
            points2 = loadData(filePath2);
            labelFileName2.Content = filePath2;
            if (points1 != null)
            {
                ComparePoints(points1, points2);
            }
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

    internal class ProjectData
    {
        public string Label;
        public string Path;
        public override string ToString()
        {
            return Label;
        }
    }
}
