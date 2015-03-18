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
        private Point[] _points1, _points2;
        private readonly String _path;

        public MainWindow()
        {
            _path = "..\\..\\..\\..\\";

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

            for (int i = 2; i <= 256; i = i <= 32 ? i*2 : i + 32)
            {
                ComboBoxDimension.Items.Add(i.ToString());
            }
            for (int i = 2; i <= 32; i *= 2)
            {
                ComboBoxCluster.Items.Add(i.ToString());
            }
            for (int i = 8096; i <= 1048576; i *= 2)
            {
                ComboBoxSize.Items.Add(string.Format("{0}K", i/1000));
            }
            ComboBoxDataType.Items.Add("Double");
            ComboBoxDataType.Items.Add("Float");
            ComboBoxDistribution.Items.Add("Normal");
            ComboBoxDistribution.Items.Add("Uniform");

            ComboBoxCluster.SelectedIndex = 0;
            ComboBoxDataType.SelectedIndex = 0;
            ComboBoxDimension.SelectedIndex = 0;
            ComboBoxDistribution.SelectedIndex = 0;
            ComboBoxSize.SelectedIndex = 0;
        }

        private void buttonLoadFiles1_Click(object sender, RoutedEventArgs e)
        {
            var ofd = new OpenFileDialog {DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat"};

            bool? result = ofd.ShowDialog();


            if (result != null && result.Value)
            {
                _points1 = loadData(ofd.FileName);
                labelFileName1.Content = ofd.FileName;
                if (_points2 != null)
                {
                    ComparePoints(_points1, _points2);
                }
            }
        }

        private void buttonLoadFiles2_Click(object sender, RoutedEventArgs e)
        {
            var ofd = new OpenFileDialog {DefaultExt = ".dat", Filter = "Data files (*.dat)|*.dat"};

            bool? result = ofd.ShowDialog();


            if (result != null && result.Value)
            {
                _points2 = loadData(ofd.FileName);
                labelFileName2.Content = ofd.FileName;
                if (_points1 != null)
                {
                    ComparePoints(_points1, _points2);
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
                listBoxDifferentFiles.Items.Add("Files cointains different number of points (" + a.Length + " : " +
                                                b.Length + ")");
                ++differencesCount;
            }
            else
            {
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i].Cluster != b[i].Cluster)
                    {
                        listBoxDifferentFiles.Items.Add("Points " + i + " are assigned to different cluster (" +
                                                        a[i].Cluster + " : " + b[i].Cluster +
                                                        ") distances from cluster: " + a[i].DistanceFromCluster + " " +
                                                        b[i].DistanceFromCluster);
                        ++differencesCount;
                    }
                    else
                    {
                        for (int j = 0; j < a[i].Coords.Length; j++)
                        {
                            if (Double.IsNaN(a[i].Coords[j]) || Double.IsNaN(b[i].Coords[j]) ||
                                (Math.Abs(a[i].Coords[j] - b[i].Coords[j]) > 0.0001))
                            {
                                listBoxDifferentFiles.Items.Add("Points " + i +
                                                                " have different coordinate (coordinate: " + j +
                                                                ", values: " + a[i].Coords[j] + " : " + b[i].Coords[j] +
                                                                ")");
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
            _points1 = loadData(labelFileName1.Content.ToString());
            _points2 = loadData(labelFileName2.Content.ToString());
            ComparePoints(_points1, _points2);
        }

        private void ButtonSwitch_OnClick(object sender, RoutedEventArgs e)
        {
            string file1 = labelFileName1.Content.ToString();
            string file2 = labelFileName2.Content.ToString();
            if (!file1.Contains("clusters"))
            {
                if (file1.Length > file1.LastIndexOf("means", StringComparison.Ordinal))
                    file1 = file1.Insert(file1.LastIndexOf("means", StringComparison.Ordinal), "clusters");
                if (file1.Length > file1.LastIndexOf("means", StringComparison.Ordinal))
                    file1 = file1.Remove(file1.LastIndexOf("means", StringComparison.Ordinal), 5);
                if (file2.Length > file2.LastIndexOf("means", StringComparison.Ordinal))
                    file2 = file2.Insert(file2.LastIndexOf("means", StringComparison.Ordinal), "clusters");
                if (file2.Length > file2.LastIndexOf("means", StringComparison.Ordinal))
                    file2 = file2.Remove(file2.LastIndexOf("means", StringComparison.Ordinal), 5);
            }
            else
            {
                file1 = file1.Replace("clusters", "means");
                file2 = file2.Replace("clusters", "means");
            }

            _points1 = loadData(file1);
            _points2 = loadData(file2);
            labelFileName1.Content = file1;
            labelFileName2.Content = file2;
            ComparePoints(_points1, _points2);
        }

        private void ComboBoxLocation1_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            string type = labelFileName2.Content.ToString().Contains("clusters") ? "clusters" : "means";
            string filePath1 = string.Format("{0}{1}{2}{3}{4}{5}D{6}{7}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path, type,
                     ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxSize.SelectedItem, ComboBoxCluster.SelectedItem);
            
            _points1 = loadData(filePath1);
            labelFileName1.Content = filePath1;
            if (_points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxLocation2_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            string type = labelFileName1.Content.ToString().Contains("clusters") ? "clusters" : "means";

            string filePath2 = string.Format("{0}{1}{2}{3}{4}{5}D{6}{7}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path, type,
                     ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxSize.SelectedItem, ComboBoxCluster.SelectedItem);
            _points2 = loadData(filePath2);
            labelFileName2.Content = filePath2;
            if (_points1 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxDimension_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxLocation1.SelectedItem != null && CheckSelectionBox())
            {
                string filePath1 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points1 = loadData(filePath1);
                labelFileName1.Content = filePath1;
            }

            if (ComboBoxLocation2.SelectedItem != null && CheckSelectionBox())
            {
                string filePath2 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation2.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points2 = loadData(filePath2);
                labelFileName2.Content = filePath2;
            }

            if (_points1 != null && _points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxSize_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxLocation1.SelectedItem != null && CheckSelectionBox())
            {
                string filePath1 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, e.AddedItems[0], ComboBoxCluster.SelectedItem);
                _points1 = loadData(filePath1);
                labelFileName1.Content = filePath1;
            }

            if (ComboBoxLocation2.SelectedItem != null && CheckSelectionBox())
            {
                string filePath2 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData) ComboBoxLocation2.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, e.AddedItems[0], ComboBoxCluster.SelectedItem);
                _points2 = loadData(filePath2);
                labelFileName2.Content = filePath2;
            }

            if (_points1 != null && _points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxCluster_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxLocation1.SelectedItem != null && CheckSelectionBox())
            {
                string filePath1 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, e.AddedItems[0]);
                _points1 = loadData(filePath1);
                labelFileName1.Content = filePath1;
            }

            if (ComboBoxLocation2.SelectedItem != null && CheckSelectionBox())
            {
                string filePath2 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation2.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, e.AddedItems[0]);
                _points2 = loadData(filePath2);
                labelFileName2.Content = filePath2;
            }

            if (_points1 != null && _points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxDistribution_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxLocation1.SelectedItem != null && CheckSelectionBox())
            {
                string filePath1 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], e.AddedItems[0].ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points1 = loadData(filePath1);
                labelFileName1.Content = filePath1;
            }

            if (ComboBoxLocation2.SelectedItem != null && CheckSelectionBox())
            {
                string filePath2 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation2.SelectedItem).Path,
                    ComboBoxDataType.SelectedItem.ToString()[0], e.AddedItems[0].ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points2 = loadData(filePath2);
                labelFileName2.Content = filePath2;
            }

            if (_points1 != null && _points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private void ComboBoxDataType_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxLocation1.SelectedItem != null && CheckSelectionBox())
            {
                string filePath1 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation1.SelectedItem).Path,
                    e.AddedItems[0].ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points1 = loadData(filePath1);
                labelFileName1.Content = filePath1;
            }

            if (ComboBoxLocation2.SelectedItem != null && CheckSelectionBox())
            {
                string filePath2 = string.Format("{0}{1}clusters{2}{3}{4}D{5}{6}C.dat", _path, ((ProjectData)ComboBoxLocation2.SelectedItem).Path,
                    e.AddedItems[0].ToString()[0], ComboBoxDistribution.SelectedItem.ToString()[0], ComboBoxDimension.SelectedItem, ComboBoxDimension.SelectedItem, ComboBoxCluster.SelectedItem);
                _points2 = loadData(filePath2);
                labelFileName2.Content = filePath2;
            }

            if (_points1 != null && _points2 != null)
            {
                ComparePoints(_points1, _points2);
            }
        }

        private bool CheckSelectionBox()
        {
            return (ComboBoxCluster.SelectedIndex != -1)
                && (ComboBoxDataType.SelectedIndex != -1)
                && (ComboBoxDimension.SelectedIndex != -1)
                && (ComboBoxDistribution.SelectedIndex != -1)
                && (ComboBoxSize.SelectedIndex != -1);
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
