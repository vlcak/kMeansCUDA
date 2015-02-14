﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.IO;
using System.Diagnostics;
using OpenTK;
using Edge = System.Collections.Generic.KeyValuePair<int, int>;

namespace Scene3D
{
    /// <summary>
    /// B-rep 3D scene with associated corner-table (Jarek Rossignac).
    /// </summary>
    public partial class SceneBrep
    {
        #region Constants

        /// <summary>
        /// Invalid handle (for vertices, trinagles, corners..).
        /// </summary>
        public const int NULL = -1;

        #endregion

        #region Scene data

        /// <summary>
        /// Array of vertex coordinates (float[3]).
        /// </summary>
        protected List<Vector3> geometry = null;

        /// <summary>
        /// Array of normal vectors (non mandatory).
        /// </summary>
        protected List<Vector3> normals = null;

        /// <summary>
        /// Array of vertex colors (non mandatory).
        /// </summary>
        protected List<Vector3> colors = null;

        /// <summary>
        /// Array of 2D texture coordinates (non mandatory).
        /// </summary>
        protected List<Vector2> txtCoords = null;

        /// <summary>
        /// Vertex pointer (handle) for each corner.
        /// </summary>
        protected List<int> vertexPtr = null;

        /// <summary>
        /// Opposite corner pointer (handle) for each corner.
        /// Valid only for topological scene (triangles are connected).
        /// </summary>
        protected List<int> oppositePtr = null;

        public int statEdges = 0;
        public int statShared = 0;

        #endregion

        #region Construction

        public SceneBrep()
        {
            Reset();
        }

        #endregion

        #region B-rep API

        /// <summary>
        /// [Re]initializes the scene data.
        /// </summary>
        public void Reset()
        {
            geometry = new List<Vector3>(256);
            normals = null;
            colors = null;
            txtCoords = null;
            vertexPtr = new List<int>(256);
            oppositePtr = null;
        }

        /// <summary>
        /// Current number of vertices in the scene.
        /// </summary>
        public int Vertices
        {
            get
            {
                return (geometry == null) ? 0 : geometry.Count;
            }
        }

        /// <summary>
        /// Current number of normal vectors in the scene (should be 0 or the same as Vertices).
        /// </summary>
        public int Normals
        {
            get
            {
                return (normals == null) ? 0 : normals.Count;
            }
        }

        public bool HasNormals()
        {
            return (normals != null);
        }

        public int NormalBytes()
        {
            return ((normals != null) ? 3 * sizeof(float) : 0);
        }

        /// <summary>
        /// Current number of vertex colors in the scene (should be 0 or the same as Vertices).
        /// </summary>
        public int Colors
        {
            get
            {
                return (colors == null) ? 0 : colors.Count;
            }
        }

        public bool HasColors()
        {
            return (colors != null);
        }

        public int ColorBytes()
        {
            return ((colors != null) ? 3 * sizeof(float) : 0);
        }

        /// <summary>
        /// Current number of texture coordinates in the scene (should be 0 or the same as Vertices).
        /// </summary>
        public int TxtCoords
        {
            get
            {
                return (txtCoords == null) ? 0 : txtCoords.Count;
            }
        }

        public bool HasTxtCoords()
        {
            return (txtCoords != null);
        }

        public int TxtCoordsBytes()
        {
            return ((txtCoords != null) ? 2 * sizeof(float) : 0);
        }

        /// <summary>
        /// Current number of triangles in the scene.
        /// </summary>
        public int Triangles
        {
            get
            {
                if (vertexPtr == null) return 0;
                Debug.Assert(vertexPtr.Count % 3 == 0, "Invalid V[] size");
                return vertexPtr.Count / 3;
            }
        }

        /// <summary>
        /// Current number of corners in the scene (# of triangles times three).
        /// </summary>
        public int Corners
        {
            get
            {
                return (vertexPtr == null) ? 0 : vertexPtr.Count;
            }
        }

        /// <summary>
        /// Add a new vertex defined by its 3D coordinate.
        /// </summary>
        /// <param name="v">Vertex coordinate in the object space</param>
        /// <returns>Vertex handle</returns>
        public int AddVertex(Vector3 v)
        {
            Debug.Assert(geometry != null);

            int handle = geometry.Count;
            geometry.Add(v);

            if (normals != null)
            {
                Debug.Assert(normals.Count == handle, "Invalid N[] size");
                normals.Add(Vector3.UnitY);
            }

            if (colors != null)
            {
                Debug.Assert(colors.Count == handle, "Invalid C[] size");
                colors.Add(Vector3.One);
            }

            if (txtCoords != null)
            {
                Debug.Assert(txtCoords.Count == handle, "Invalid T[] size");
                txtCoords.Add(Vector2.Zero);
            }

            return handle;
        }

        /// <summary>
        /// Returns object-space coordinates of the given vertex.
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <returns>Object-space coordinates</returns>
        public Vector3 GetVertex(int v)
        {
            Debug.Assert(geometry != null, "Invalid G[]");
            Debug.Assert(0 <= v && v < geometry.Count, "Invalid vertex handle");
            return geometry[v];
        }

        /// <summary>
        /// Assigns a normal vector to an existing vertex
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <param name="normal">New normal vector</param>
        public void SetNormal(int v, Vector3 normal)
        {
            Debug.Assert(geometry != null, "Invalid G[]");
            Debug.Assert(0 <= v && v < geometry.Count, "Invalid vertex handle");

            if (normals == null)
            {
                normals = new List<Vector3>(geometry.Count);
                for (int i = 0; i < geometry.Count; i++)
                    normals.Add(Vector3.UnitX);
            }

            normals[v] = normal;
        }

        /// <summary>
        /// Returns normal vector of the given vertex.
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <returns>Normal vector</returns>
        public Vector3 GetNormal(int v)
        {
            Debug.Assert(normals != null, "Invalid N[]");
            Debug.Assert(0 <= v && v < normals.Count, "Invalid vertex handle");
            return normals[v];
        }

        /// <summary>
        /// Assigns a color to an existing vertex
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <param name="color">New vertex color</param>
        public void SetColor(int v, Vector3 color)
        {
            Debug.Assert(geometry != null, "Invalid G[]");
            Debug.Assert(0 <= v && v < geometry.Count, "Invalid vertex handle");

            if (colors == null)
            {
                colors = new List<Vector3>(geometry.Count);
                for (int i = 0; i < geometry.Count; i++)
                    colors.Add(Vector3.One);
            }

            colors[v] = color;
        }

        /// <summary>
        /// Returns color of the given vertex.
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <returns>Vertex color</returns>
        public Vector3 GetColor(int v)
        {
            Debug.Assert(colors != null, "Invalid C[]");
            Debug.Assert(0 <= v && v < colors.Count, "Invalid vertex handle");
            return colors[v];
        }

        /// <summary>
        /// Assigns a texture coordinate to an existing vertex
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <param name="txt">New texture coordinate</param>
        public void SetTxtCoord(int v, Vector2 txt)
        {
            Debug.Assert(geometry != null, "Invalid G[]");
            Debug.Assert(0 <= v && v < geometry.Count, "Invalid vertex handle");

            if (txtCoords == null)
            {
                txtCoords = new List<Vector2>(geometry.Count);
                for (int i = 0; i < geometry.Count; i++)
                    txtCoords.Add(Vector2.Zero);
            }

            txtCoords[v] = txt;
        }

        /// <summary>
        /// Returns texture coordinate of the given vertex.
        /// </summary>
        /// <param name="v">Vertex handle</param>
        /// <returns>Texture coordinate</returns>
        public Vector2 GetTxtCoord(int v)
        {
            Debug.Assert(txtCoords != null, "Invalid T[]");
            Debug.Assert(0 <= v && v < txtCoords.Count, "Invalid vertex handle");
            return txtCoords[v];
        }

        /// <summary>
        /// Adds a new triangle face defined by its vertices.
        /// </summary>
        /// <param name="v1">Handle of the 1st vertex</param>
        /// <param name="v2">Handle of the 2nd vertex</param>
        /// <param name="v3">Handle of the 3rd vertex</param>
        /// <returns>Triangle handle</returns>
        public int AddTriangle(int v1, int v2, int v3)
        {
            Debug.Assert(geometry != null, "Invalid G[] size");
            Debug.Assert(geometry.Count > v1 &&
                          geometry.Count > v2 &&
                          geometry.Count > v3, "Invalid vertex handle");
            Debug.Assert(vertexPtr != null && (vertexPtr.Count % 3 == 0),
                          "Invalid corner-table (V[] size)");

            int handle1 = vertexPtr.Count;
            vertexPtr.Add(v1);
            vertexPtr.Add(v2);
            vertexPtr.Add(v3);

            if (oppositePtr != null)
            {
                Debug.Assert(oppositePtr.Count == handle1, "Invalid O[] size");
                oppositePtr.Add(NULL);
                oppositePtr.Add(NULL);
                oppositePtr.Add(NULL);
            }

            return handle1 / 3;
        }

        /// <summary>
        /// Returns vertex handles of the given triangle.
        /// </summary>
        /// <param name="tr">Triangle handle</param>
        /// <param name="v1">Variable to receive the 1st vertex handle</param>
        /// <param name="v2">Variable to receive the 2nd vertex handle</param>
        /// <param name="v3">Variable to receive the 3rd vertex handle</param>
        public void GetTriangleVertices(int tr, out int v1, out int v2, out int v3)
        {
            Debug.Assert(geometry != null, "Invalid G[] size");
            tr *= 3;
            Debug.Assert(vertexPtr != null && 0 <= tr && tr + 2 < vertexPtr.Count,
                          "Invalid triangle handle");

            v1 = vertexPtr[tr];
            v2 = vertexPtr[tr + 1];
            v3 = vertexPtr[tr + 2];
        }

        /// <summary>
        /// Returns vertex coordinates of the given triangle.
        /// </summary>
        /// <param name="tr">Triangle handle</param>
        /// <param name="v1">Variable to receive the 1st vertex coordinates</param>
        /// <param name="v2">Variable to receive the 2nd vertex coordinates</param>
        /// <param name="v3">Variable to receive the 3rd vertex coordinates</param>
        public void GetTriangleVertices(int tr, out Vector3 v1, out Vector3 v2, out Vector3 v3)
        {
            Debug.Assert(geometry != null, "Invalid G[] size");
            tr *= 3;
            Debug.Assert(vertexPtr != null && 0 <= tr && tr + 2 < vertexPtr.Count,
                          "Invalid triangle handle");

            int h1 = vertexPtr[tr];
            int h2 = vertexPtr[tr + 1];
            int h3 = vertexPtr[tr + 2];
            v1 = (h1 < 0 || h1 >= geometry.Count) ? Vector3.Zero : geometry[h1];
            v2 = (h2 < 0 || h2 >= geometry.Count) ? Vector3.Zero : geometry[h2];
            v3 = (h3 < 0 || h3 >= geometry.Count) ? Vector3.Zero : geometry[h3];
        }

        /// <summary>
        /// Returns vertex coordinates of the given triangle.
        /// </summary>
        /// <param name="tr">Triangle handle</param>
        /// <param name="v1">Variable to receive the 1st vertex coordinates</param>
        /// <param name="v2">Variable to receive the 2nd vertex coordinates</param>
        /// <param name="v3">Variable to receive the 3rd vertex coordinates</param>
        public void GetTriangleVertices(int tr, out Vector4 v1, out Vector4 v2, out Vector4 v3)
        {
            Debug.Assert(geometry != null, "Invalid G[] size");
            tr *= 3;
            Debug.Assert(vertexPtr != null && 0 <= tr && tr + 2 < vertexPtr.Count,
                          "Invalid triangle handle");

            int h1 = vertexPtr[tr];
            int h2 = vertexPtr[tr + 1];
            int h3 = vertexPtr[tr + 2];
            v1 = new Vector4((h1 < 0 || h1 >= geometry.Count) ? Vector3.Zero : geometry[h1], 1.0f);
            v2 = new Vector4((h2 < 0 || h2 >= geometry.Count) ? Vector3.Zero : geometry[h2], 1.0f);
            v3 = new Vector4((h3 < 0 || h3 >= geometry.Count) ? Vector3.Zero : geometry[h3], 1.0f);
        }

        /// <summary>
        /// Computes vertex array size (VBO) in bytes.
        /// </summary>
        /// <param name="vertices">Use vertex coordinates?</param>
        /// <param name="txt">Use texture coordinates?</param>
        /// <param name="col">Use vertex colors?</param>
        /// <param name="norm">Use normal vectors?</param>
        /// <returns>Buffer size in bytes</returns>
        public int VertexBufferSize(bool vertices, bool txt, bool col, bool norm)
        {
            Debug.Assert(geometry != null, "Invalid G[]");

            int size = 0;
            if (vertices)
                size += Vertices * 3 * sizeof(float);
            if (txt && TxtCoords > 0)
                size += Vertices * 2 * sizeof(float);
            if (col && Colors > 0)
                size += Vertices * 3 * sizeof(float);
            if (norm && Normals > 0)
                size += Vertices * 3 * sizeof(float);

            return size;
        }

        /// <summary>
        /// Fill vertex data into the provided memory array (VBO after MapBuffer).
        /// </summary>
        /// <param name="ptr">Memory pointer</param>
        /// <param name="vertices">Use vertex coordinates?</param>
        /// <param name="txt">Use texture coordinates?</param>
        /// <param name="col">Use vertex colors?</param>
        /// <param name="norm">Use normal vectors?</param>
        /// <returns>Stride (vertex size) in bytes</returns>
        public unsafe int FillVertexBuffer(float* ptr, bool vertices, bool txt, bool col, bool norm)
        {
            if (geometry == null) return 0;

            if (txt && TxtCoords < Vertices)
                txt = false;

            if (col && Colors < Vertices)
                col = false;

            if (norm && Normals < Vertices)
                norm = false;

            int i;
            for (i = 0; i < Vertices; i++)
            {
                // GL_T2F_C3F_N3F_V3F

                if (txt)
                {
                    *ptr++ = txtCoords[i].X;
                    *ptr++ = txtCoords[i].Y;
                }
                if (col)
                {
                    *ptr++ = colors[i].X;
                    *ptr++ = colors[i].Y;
                    *ptr++ = colors[i].Z;
                }
                if (norm)
                {
                    *ptr++ = normals[i].X;
                    *ptr++ = normals[i].Y;
                    *ptr++ = normals[i].Z;
                }
                if (vertices)
                {
                    *ptr++ = geometry[i].X;
                    *ptr++ = geometry[i].Y;
                    *ptr++ = geometry[i].Z;
                }
            }

            return sizeof(float) * ((txt ? 2 : 0) + (col ? 3 : 0) + (norm ? 3 : 0) + (vertices ? 3 : 0));
        }

        /// <summary>
        /// Fills index data into provided memory array (VBO after MapBuffer).
        /// </summary>
        /// <param name="ptr">Memory pointer</param>
        public unsafe void FillIndexBuffer(uint* ptr)
        {
            if (vertexPtr == null) return;

            foreach (int i in vertexPtr)
                *ptr++ = (uint)i;
        }

        /// <summary>
        /// Computes center point and diameter of the scene.
        /// </summary>
        /// <param name="center">Center point</param>
        /// <returns>Diameter</returns>
        public float GetDiameter(out Vector3 center)
        {
            // center of the object = point to look at:
            double cx = 0.0;
            double cy = 0.0;
            double cz = 0.0;
            float minx = float.MaxValue;
            float miny = float.MaxValue;
            float minz = float.MaxValue;
            float maxx = float.MinValue;
            float maxy = float.MinValue;
            float maxz = float.MinValue;
            int i;

            for (i = 0; i < Vertices; i++)
            {
                Vector3 vi = GetVertex(i);
                cx += vi.X;
                cy += vi.Y;
                cz += vi.Z;
                if (vi.X < minx) minx = vi.X;
                if (vi.Y < miny) miny = vi.Y;
                if (vi.Z < minz) minz = vi.Z;
                if (vi.X > maxx) maxx = vi.X;
                if (vi.Y > maxy) maxy = vi.Y;
                if (vi.Z > maxz) maxz = vi.Z;
            }
            center = new Vector3((float)(cx / Vertices),
                                  (float)(cy / Vertices),
                                  (float)(cz / Vertices));
            return (float)Math.Sqrt((maxx - minx) * (maxx - minx) +
                                     (maxy - miny) * (maxy - miny) +
                                     (maxz - minz) * (maxz - minz));
        }

        /// <summary>
        /// Generate random vertex colors.
        /// </summary>
        /// <param name="seed">Random seed</param>
        /// <param name="clusters">array with assigned clusters</param>
        public void GenerateColors(int seed, int[] clusters)
        {
            Random rnd = new Random(seed);


            Color[] myColors = new[]
            {
                Color.Pink,
                Color.DeepPink,
                Color.Red,
                Color.OrangeRed,
                Color.Orange,
                Color.Green,
                Color.Lime,
                Color.GreenYellow,
                Color.MediumSpringGreen,
                Color.DarkOliveGreen,
                Color.Blue,
                Color.Aqua,
                Color.MediumTurquoise,
                Color.Purple,
                Color.MediumPurple,
                Color.Yellow,
                Color.White,
                Color.Black,
                Color.Gray,
                Color.DarkGray,
                Color.Chocolate,
                Color.Coral,
                Color.Crimson,
                Color.DarkMagenta,
                Color.DarkSeaGreen,
                Color.DodgerBlue,
                Color.Fuchsia,
                Color.Gold,
                Color.Goldenrod, 
                Color.Indigo,
                Color.OliveDrab,
                Color.Teal
            };

            if (colors == null || Colors < Vertices)
            {
                colors = new List<Vector3>(geometry.Count);
                Vector3[] color = new Vector3[] { new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1) };
                for (int i = 0; i < geometry.Count; i++)
                    //if (i % 8 == 0)
                    colors.Add(new Vector3(myColors[clusters[i / 8]].R / 255f, myColors[clusters[i / 8]].G / 255f, myColors[clusters[i / 8]].B / 255f));
            }
            else
                for (int i = 0; i < geometry.Count; i++)
                    colors[i] = new Vector3((float)rnd.NextDouble(), (float)rnd.NextDouble(), (float)rnd.NextDouble());
        }

        #endregion

        #region Corner-table API

        /// <summary>
        /// [Re]builds the mesh topology (corner-table should be consistent after this call).
        /// </summary>
        public void BuildCornerTable()
        {
            if (geometry == null || geometry.Count < 1 ||
                 vertexPtr == null || vertexPtr.Count < 1)
            {
                Reset();
                return;
            }

            int n = vertexPtr.Count;
            oppositePtr = new List<int>(n);
            for (int i = 0; i < n; i++)
                oppositePtr.Add(NULL);
            Dictionary<Edge, int> edges = new Dictionary<Edge, int>();

            statEdges = statShared = 0;
            for (int i = 0; i < n; i++)               // process one corner
            {
                int cmin = cVertex(cPrev(i));
                int cmax = cVertex(cNext(i));
                if (cmin < 0 || cmax < 0) continue;

                if (cmin > cmax)
                {
                    int tmp = cmin;
                    cmin = cmax;
                    cmax = tmp;
                }
                Edge edge = new Edge(cmin, cmax);
                if (edges.ContainsKey(edge))
                {
                    int other = edges[edge];
                    Debug.Assert(oppositePtr[other] == NULL);
                    oppositePtr[other] = i;
                    oppositePtr[i] = other;
                    edges.Remove(edge);
                    statShared++;
                }
                else
                {
                    edges.Add(edge, i);
                    statEdges++;
                }
            }
        }

        /// <summary>
        /// Returns triangle handle of the given corner
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Triangle handle</returns>
        public static int cTriangle(int c)
        {
            return c / 3;
        }

        /// <summary>
        /// Returns handle of the 1st corner of the given triangle
        /// </summary>
        /// <param name="tr">Triangle handle</param>
        /// <returns>Corner handle</returns>
        public static int tCorner(int tr)
        {
            return tr * 3;
        }

        /// <summary>
        /// Returns the next corner inside the same triangle
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Handle of the next corner</returns>
        public static int cNext(int c)
        {
            return (c % 3 == 2) ? c - 2 : c + 1;
        }

        /// <summary>
        /// Returns the previous corner inside the same triangle
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Handle of the previous corner</returns>
        public static int cPrev(int c)
        {
            return (c % 3 == 0) ? c + 2 : c - 1;
        }

        /// <summary>
        /// Returns vertex handle of the given corner
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Associated vertex's handle</returns>
        public int cVertex(int c)
        {
            if (c < 0) return NULL;

            Debug.Assert(vertexPtr != null, "Invalid V[] array");
            Debug.Assert(c < vertexPtr.Count, "Invalid corner handle");

            return vertexPtr[c];
        }

        /// <summary>
        /// Returns opposite corner to the given corner
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Handle of the opposite corner</returns>
        public int cOpposite(int c)
        {
            if (c < 0) return NULL;

            Debug.Assert(oppositePtr != null, "Invalid O[] array");
            Debug.Assert(c < oppositePtr.Count, "Invalid corner handle");

            return oppositePtr[c];
        }

        /// <summary>
        /// Returns the "right" corner from the given corner
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Corner handle of the "right" triangle</returns>
        public int cRight(int c)
        {
            return cOpposite(cNext(c));
        }

        /// <summary>
        /// Returns the "left" corner from the given corner
        /// </summary>
        /// <param name="c">Corner handle</param>
        /// <returns>Corner handle of the "left" triangle</returns>
        public int cLeft(int c)
        {
            return cOpposite(cPrev(c));
        }

        #endregion
    }
}
