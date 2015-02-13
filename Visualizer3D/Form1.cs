using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Scene3D;

namespace _038trackball
{
  public partial class Form1 : Form
  {
    /// <summary>
    /// Scene read from file.
    /// </summary>
    protected SceneBrep scene = new SceneBrep();

    /// <summary>
    /// Scene center point.
    /// </summary>
    protected Vector3 center = Vector3.Zero;

    /// <summary>
    /// Scene diameter.
    /// </summary>
    protected float diameter = 3.5f;

    /// <summary>
    /// GLControl guard flag.
    /// </summary>
    bool loaded = false;

    /// <summary>
    /// Are we allowed to use VBO?
    /// </summary>
    bool useVBO = true;

    #region OpenGL globals

    private uint[] VBOid = new uint[ 2 ];       // vertex array (colors, normals, coords), index array
    private int stride = 0;                     // stride for vertex array

    #endregion

    #region FPS counter

    long lastFpsTime = 0L;
    int frameCounter = 0;
    long triangleCounter = 0L;

    #endregion

    public Form1 ()
    {
      InitializeComponent();
    }
    
    private void glControl1_Load ( object sender, EventArgs e )
    {
      loaded = true;

      // OpenGL init code:
      GL.ClearColor( Color.DarkBlue );
      GL.Enable( EnableCap.DepthTest );
      GL.ShadeModel( ShadingModel.Flat );

      // VBO init:
      GL.GenBuffers( 2, VBOid );
      if ( GL.GetError() != ErrorCode.NoError )
        useVBO = false;

      SetupViewport();

      Application.Idle += new EventHandler( Application_Idle );      
      comboTrackballType.SelectedIndex = 0;
    }

    private void glControl1_Resize ( object sender, EventArgs e )
    {
      if ( !loaded ) return;

      SetupViewport();
      glControl1.Invalidate();
    }

    private void glControl1_Paint ( object sender, PaintEventArgs e )
    {
      Render();
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
                        var p = new Point(3);
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

    private void buttonOpen_Click ( object sender, EventArgs e )
    {
      OpenFileDialog ofd = new OpenFileDialog();

      ofd.Title = "Open Scene File";
      //ofd.Filter = "Wavefront OBJ Files|*.obj" +
      //    "|All scene types|*.obj";

        ofd.Filter = "Data files (*.dat)|*.dat";

      ofd.FilterIndex = 1;
      ofd.FileName = "";
      if ( ofd.ShowDialog() != DialogResult.OK )
        return;

       Point[] points = loadData(ofd.FileName);
        createWaveformFile(points);


      WavefrontObj objReader = new WavefrontObj();
      objReader.MirrorConversion = false;
      StreamReader reader = new StreamReader( new FileStream( "temp.obj", FileMode.Open ) );
      int faces = objReader.ReadBrep( reader, scene );
      reader.Close();
      scene.BuildCornerTable();
      diameter = scene.GetDiameter( out center );
      scene.GenerateColors( 12, points.Select(p => p.Cluster).ToArray() );
      ResetCamera();

      labelFile.Text = String.Format( "{0}: {1} faces", ofd.SafeFileName, faces );
      PrepareDataBuffers();
      glControl1.Invalidate();
    }

    /// <summary>
    /// Prepare VBO content and upload it to the GPU.
    /// </summary>
    private void PrepareDataBuffers ()
    {
      if ( useVBO &&
           scene != null &&
           scene.Triangles > 0 )
      {
        GL.EnableClientState( ArrayCap.VertexArray );
        if ( scene.Normals > 0 )
          GL.EnableClientState( ArrayCap.NormalArray );
        GL.EnableClientState( ArrayCap.ColorArray );

        // Vertex array: color [normal] coord
        GL.BindBuffer( BufferTarget.ArrayBuffer, VBOid[ 0 ] );
        int vertexBufferSize = scene.VertexBufferSize( true, false, true, true );
        GL.BufferData( BufferTarget.ArrayBuffer, (IntPtr)vertexBufferSize, IntPtr.Zero, BufferUsageHint.StaticDraw );
        IntPtr videoMemoryPtr = GL.MapBuffer( BufferTarget.ArrayBuffer, BufferAccess.WriteOnly );
        unsafe
        {
          stride = scene.FillVertexBuffer( (float*)videoMemoryPtr.ToPointer(), true, false, true, true );
        }
        GL.UnmapBuffer( BufferTarget.ArrayBuffer );
        GL.BindBuffer( BufferTarget.ArrayBuffer, 0 );

        // Index buffer
        GL.BindBuffer( BufferTarget.ElementArrayBuffer, VBOid[ 1 ] );
        GL.BufferData( BufferTarget.ElementArrayBuffer, (IntPtr)(scene.Triangles * 3 * sizeof( uint )), IntPtr.Zero, BufferUsageHint.StaticDraw );
        videoMemoryPtr = GL.MapBuffer( BufferTarget.ElementArrayBuffer, BufferAccess.WriteOnly );
        unsafe
        {
          scene.FillIndexBuffer( (uint*)videoMemoryPtr.ToPointer() );
        }
        GL.UnmapBuffer( BufferTarget.ElementArrayBuffer );
        GL.BindBuffer( BufferTarget.ElementArrayBuffer, 0 );
      }
      else
      {
        GL.DisableClientState( ArrayCap.VertexArray );
        GL.DisableClientState( ArrayCap.NormalArray );
        GL.DisableClientState( ArrayCap.ColorArray );

        if ( useVBO )
        {
          GL.BindBuffer( BufferTarget.ArrayBuffer, VBOid[ 0 ] );
          GL.BufferData( BufferTarget.ArrayBuffer, (IntPtr)0, IntPtr.Zero, BufferUsageHint.StaticDraw );
          GL.BindBuffer( BufferTarget.ArrayBuffer, 0 );
          GL.BindBuffer( BufferTarget.ElementArrayBuffer, VBOid[ 1 ] );
          GL.BufferData( BufferTarget.ElementArrayBuffer, (IntPtr)0, IntPtr.Zero, BufferUsageHint.StaticDraw );
          GL.BindBuffer( BufferTarget.ElementArrayBuffer, 0 );
        }
      }
    }

    private void timer1_Tick(object sender, EventArgs e)
    {
        Vector3 axis;
        float angle;
        RotationFromCenter.ToAxisAngle(out axis, out angle);
        angle += rotationDirection * (float)numericSensitivity.Value;
        angle %= (float)(2 * Math.PI);
        RotationFromCenter = Quaternion.FromAxisAngle(axis, angle);
    }

      private void createWaveformFile(Point[] points)
      {
          using (StreamWriter sw = new System.IO.StreamWriter("temp.obj"))
          {
              Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
              sw.Write("# cube.obj\n#\n\ng cube\n\nvn 0.0 0.0 1.0\nvn 0.0 0.0 -1.0\nvn 0.0 1.0 0.0\nvn 0.0 -1.0 0.0\nvn 1.0 0.0 0.0\nvn -1.0 0.0 0.0\n\n");
              for (int i = 0; i < points.Length; i++)
              {
                  writeCube(sw, points[i], i);
              }
          }
      }

      private void writeCube(StreamWriter sw, Point p, int pointNumber)
      {
          for (int i = 0; i < 8; i++)
          {
              sw.Write("v");
              for (int j = 0; j < 3; j++)
              {
                  sw.Write(" " + (p.Coords[j] + ((i >> (2 - j)) % 2) / 10f));

              }
              sw.Write("\n");
          }
          sw.Write("\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//2 " + (pointNumber * 8 + 7) + "//2 " + (pointNumber * 8 + 5) + "//2\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//2 " + (pointNumber * 8 + 3) + "//2 " + (pointNumber * 8 + 7) + "//2\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//4 " + (pointNumber * 8 + 5) + "//4 " + (pointNumber * 8 + 6) + "//4\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//4 " + (pointNumber * 8 + 6) + "//4 " + (pointNumber * 8 + 2) + "//4\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//6 " + (pointNumber * 8 + 4) + "//6 " + (pointNumber * 8 + 3) + "//6\n");
          sw.Write("f " + (pointNumber * 8 + 1) + "//6 " + (pointNumber * 8 + 2) + "//6 " + (pointNumber * 8 + 4) + "//6\n");
          sw.Write("f " + (pointNumber * 8 + 3) + "//3 " + (pointNumber * 8 + 8) + "//3 " + (pointNumber * 8 + 7) + "//3\n");
          sw.Write("f " + (pointNumber * 8 + 3) + "//3 " + (pointNumber * 8 + 4) + "//3 " + (pointNumber * 8 + 8) + "//3\n");
          sw.Write("f " + (pointNumber * 8 + 5) + "//5 " + (pointNumber * 8 + 7) + "//5 " + (pointNumber * 8 + 8) + "//5\n");
          sw.Write("f " + (pointNumber * 8 + 5) + "//5 " + (pointNumber * 8 + 8) + "//5 " + (pointNumber * 8 + 6) + "//5\n");
          sw.Write("f " + (pointNumber * 8 + 2) + "//1 " + (pointNumber * 8 + 6) + "//7 " + (pointNumber * 8 + 8) + "//1\n");
          sw.Write("f " + (pointNumber * 8 + 2) + "//1 " + (pointNumber * 8 + 8) + "//7 " + (pointNumber * 8 + 4) + "//1\n");

          sw.Write("\n");
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
