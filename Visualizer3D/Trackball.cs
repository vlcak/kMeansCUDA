using System;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace _038trackball
{
    public partial class Form1
    {
        #region Camera attributes

        /// <summary>
        /// Current camera position.
        /// </summary>
        private Vector3 eye = new Vector3(0.0f, 0.0f, 10.0f);

        /// <summary>
        /// Current point to look at.
        /// </summary>
        private Vector3 pointAt = Vector3.Zero;

        /// <summary>
        /// Current "up" vector.
        /// </summary>
        private Vector3 up = Vector3.UnitY;

        /// <summary>
        /// Vertical field-of-view angle in radians.
        /// </summary>
        private float fov = 1.0f;

        /// <summary>
        /// Camera's far point.
        /// </summary>
        private float far = 200.0f;

        #endregion

        Vector3 initialVector = new Vector3();
        Vector3 centerVector = new Vector3(0f, 0f, 1f);
        Quaternion RotationFromCenter = Quaternion.Identity;
        Quaternion RotationToCenter = Quaternion.Identity;
        Quaternion OldRotation = Quaternion.Identity;
        bool perspectiveView = true;
        bool rotate = false;
        float rotationDirection = 0.0f;
        Matrix4 projection;
        float scale = 1f;
        /// <summary>
        /// Sets up a projective viewport
        /// </summary>
        private void SetupViewport()
        {
            int width = glControl1.Width;
            int height = glControl1.Height;

            // 1. set ViewPort transform:
            GL.Viewport(0, 0, width, height);
            // 2. set projection matrix
            GL.MatrixMode(MatrixMode.Projection);
            if (perspectiveView)
                projection = Matrix4.CreatePerspectiveFieldOfView(fov, (float)glControl1.Width / (float)glControl1.Height, 0.1f, far);
            else
                projection = Matrix4.CreateOrthographic(1.967f * glControl1.Width / glControl1.Height, 1.967f, 0.1f, far);
            GL.LoadMatrix(ref projection);
        }

        /// <summary>
        /// Setup of a camera called for every frame prior to any rendering.
        /// </summary>
        private void SetCamera()
        {
            // !!!{{ TODO: add camera setup here
            GL.MatrixMode(MatrixMode.Modelview);
            Matrix4 modelview = Matrix4.CreateTranslation(-center) *
                //Matrix4.CreateOrthographic(Width, Height, 0, -20) *
                                Matrix4.Scale(scale / diameter) *
                                Matrix4.Rotate(RotationFromCenter * RotationToCenter * OldRotation) *
                //Matrix4.Rotate(Rotation) *
                                Matrix4.CreateTranslation(0.0f, 0.0f, -1.5f);

            GL.LoadMatrix(ref modelview);

            // !!!}}
        }

        private void ResetCamera()
        {
            // !!!{{ TODO: add camera reset code here
            // !!!}}
            //Rotation = Matrix4.CreateFromAxisAngle(new Vector3(0f, 1f, 0f), 0f);
            timer1.Stop();
            RotationFromCenter = Quaternion.Identity;
            RotationToCenter = Quaternion.Identity;
            OldRotation = Quaternion.Identity;
            scale = 1f;
            perspectiveView = true;
            SetupViewport();
            rotate = false;
        }

        /// <summary>
        /// Rendering of one frame.
        /// </summary>
        private void Render()
        {
            if (!loaded) return;

            frameCounter++;
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.ShadeModel(ShadingModel.Flat);
            GL.PolygonMode(MaterialFace.Front, PolygonMode.Fill);
            GL.Enable(EnableCap.CullFace);
            drawCircle();
            //if (rotate && rotateVectorSetted)
            //{
            //    Vector3 v;
            //    float a;
            //    RotationFromCenter.ToAxisAngle(out v, out a);
            //    RotationFromCenter = new Quaternion(v, a + 0.000001f);
            //}
            SetCamera();
            RenderScene();


            glControl1.SwapBuffers();
        }

        private void glControl1_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                if (!rotate)
                {
                    Cursor.Current = Cursors.Hand;
                    Vector3 axis;
                    float angle;
                    initialVector = getVectorFromMouseAction(e);
                    Vector3.Cross(ref initialVector, ref centerVector, out axis);
                    Vector3.CalculateAngle(ref initialVector, ref centerVector, out angle);
                    if (!checkRestrict.Checked)
                    {
                        float length = (float)Math.Sqrt(initialVector.X * initialVector.X + initialVector.Y * initialVector.Y);
                        if (length > 1)
                            angle *= length;
                    }
                    angle *= (float)numericSensitivity.Value;
                    RotationToCenter = Quaternion.FromAxisAngle(axis, angle);
                    RotationFromCenter = Quaternion.FromAxisAngle(axis, -angle);
                }
                else
                {
                    OldRotation = RotationFromCenter * OldRotation;
                    RotationFromCenter = Quaternion.Identity;
                    Vector3 axis;
                    float angle;
                    initialVector = getVectorFromMouseAction(e);
                    Vector3.Cross(ref centerVector, ref initialVector, out axis);
                    Vector3.CalculateAngle(ref centerVector, ref initialVector, out angle);
                    if (angle > 0)
                        rotationDirection = 0.01f;
                    if (angle < 0)
                        rotationDirection = -0.01f;
                    RotationFromCenter = Quaternion.FromAxisAngle(axis, angle / 1000f);
                }
                //MessageBox.Show("" + e.X + ':' + e.Y + "\n" + initialVector.X + ':' + initialVector.Y + ':' + initialVector.Z);
            }
        }

        private void glControl1_MouseUp(object sender, MouseEventArgs e)
        {
            //float x = e.X / (glControl1.Width / 2f);
            //float y = e.Y / (glControl1.Height / 2f);
            //x = x - 1;
            //y = 1 - y;
            //float z = 1 - x * x - y * y > 0 ? (float)System.Math.Sqrt(1 - x * x - y * y) : 0f;
            //Vector3 TerminalVector = new Vector3(x, y, z);
            //Cursor.Current = Cursors.Default;
            //Vector3.Transform(ref TerminalVector, ref OldRotation, out TerminalVector);
            //Vector3 axis;
            //Vector3.Cross(ref initialVector, ref TerminalVector, out axis);
            //float angle;
            //Vector3.CalculateAngle(ref initialVector, ref TerminalVector, out angle);
            //angle *= (float)numericSensitivity.Value;
            //angle = -angle;
            //OldRotation = Matrix4.Invert(OldRotation);
            if (!rotate)
            {
                OldRotation = RotationFromCenter * RotationToCenter * OldRotation;
                RotationToCenter = Quaternion.Identity;
                RotationFromCenter = Quaternion.Identity;
            }
            //OldRotation =  Matrix4.Invert(OldRotation);
            // !!!{{ TODO: add the event handler here
            // !!!}}
        }

        private void glControl1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left && !rotate)
            {
                Vector3 TerminalVector = getVectorFromMouseAction(e);
                Vector3 axis;
                Vector3.Cross(ref centerVector, ref TerminalVector, out axis);
                float angle;
                Vector3.CalculateAngle(ref centerVector, ref TerminalVector, out angle);
                //if (angle > 3*Math.PI/4 )
                //{
                //    dontDraw = true;
                //    OldRotation = Rotation * OldRotation;
                //    Rotation = Quaternion.Identity;
                //    initialVector = getVectorFromMouseAction(e);
                //    dontDraw = false;
                //}
                if (!checkRestrict.Checked)
                {
                    float length = (float)Math.Sqrt(TerminalVector.X * TerminalVector.X + TerminalVector.Y * TerminalVector.Y);
                    if (length > 1)
                        angle *= length;
                }
                angle *= (float)numericSensitivity.Value;
                RotationFromCenter = Quaternion.FromAxisAngle(axis, angle);
            }
            //GL.Rotate(angle, axis);
            // !!!{{ TODO: add the event handler here
            // !!!}}
        }

        private void glControl1_MouseWheel(object sender, MouseEventArgs e)
        {
            // !!!{{ TODO: add the event handler here
            // HINT: for most mouses, e.delta / 120 is the number of wheel clicks, +/- indicated the direction
            scale *= 1 + e.Delta / 2400f;
            // !!!}}
        }

        private void glControl1_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyData)
            {
                case Keys.P:
                    perspectiveView = !perspectiveView;
                    if (perspectiveView)
                    {
                        GL.MatrixMode(MatrixMode.Projection);
                        projection = Matrix4.CreatePerspectiveFieldOfView(fov, (float)glControl1.Width / (float)glControl1.Height, 0.1f, far);
                        //proj = Matrix4.CreateOrthographic(glControl1.Width/175f, glControl1.Height/175f, 0.1f, far);
                        GL.LoadMatrix(ref projection);
                        scale /= 1.485f;
                    }
                    else
                    {
                        GL.MatrixMode(MatrixMode.Projection);
                        //projection = Matrix4.CreatePerspectiveFieldOfView(fov, (float)glControl1.Width / (float)glControl1.Height, 0.1f, far);
                        projection = Matrix4.CreateOrthographic(1.967f * glControl1.Width / glControl1.Height, 1.967f, 0.1f, far);
                        GL.LoadMatrix(ref projection);
                        scale *= 1.485f;
                    }
                    break;
                case Keys.R:
                    rotate = !rotate;
                    OldRotation = RotationFromCenter * RotationToCenter * OldRotation;
                    RotationToCenter = Quaternion.Identity;
                    RotationFromCenter = Quaternion.Identity;
                    if (rotate)
                        timer1.Start();
                    else
                    {
                        timer1.Stop();
                        rotationDirection = 0f;
                    }
                    break;
                //case Keys.W:
                //    if (perspectiveView)
                //    {
                //        dynamicFar += 10;
                //        GL.MatrixMode(MatrixMode.Projection);
                //        projection = Matrix4.CreatePerspectiveFieldOfView(fov, (float)glControl1.Width / (float)glControl1.Height, 0.1f, far + dynamicFar);
                //        //proj = Matrix4.CreateOrthographic(glControl1.Width/175f, glControl1.Height/175f, 0.1f, far);
                //        GL.LoadMatrix(ref projection);
                //        //scale /= 1.485f;
                //    }
                //    break;
                //case Keys.S:
                //    if (perspectiveView)
                //    {
                //        if (dynamicFar > 10)
                //            dynamicFar -= 10;
                //        GL.MatrixMode(MatrixMode.Projection);
                //        projection = Matrix4.CreatePerspectiveFieldOfView(fov, (float)glControl1.Width / (float)glControl1.Height, 0.1f, far + dynamicFar);
                //        //proj = Matrix4.CreateOrthographic(glControl1.Width/175f, glControl1.Height/175f, 0.1f, far);
                //        GL.LoadMatrix(ref projection);
                //        //scale /= 1.485f;
                //    }
                //    break;
            }
            // !!!{{ TODO: add the event handler here
            // !!!}}
        }

        private void glControl1_KeyUp(object sender, KeyEventArgs e)
        {
            // !!!{{ TODO: add the event handler here
            // !!!}}
        }

        private void buttonReset_Click(object sender, EventArgs e)
        {
            // !!!{{ TODO: add the event handler here

            ResetCamera();

            // !!!}}
        }

        private Vector3 getVectorFromMouseAction(MouseEventArgs e)
        {
            float x, y;
            if (comboTrackballType.SelectedIndex == 0)
            {
                x = (e.X - glControl1.Width / 2f) / (Math.Min(glControl1.Width, glControl1.Height) / 2f);
                y = (glControl1.Height / 2f - e.Y) / (Math.Min(glControl1.Width, glControl1.Height) / 2f);
            }
            else
            {
                x = e.X / (glControl1.Width / 2f);
                y = e.Y / (glControl1.Height / 2f);
                x = x - 1;
                y = 1 - y;
            }
            float z;
            if (1 - x * x - y * y > 0)
                z = (float)System.Math.Sqrt(1 - x * x - y * y);
            else
            {
                z = 0;
                float length = (float)Math.Sqrt(x * x + y * y);
                if (checkRestrict.Checked)
                {
                    x /= length;
                    y /= length;
                }
            }
            Vector3 v = new Vector3(x, y, z);
            return v;
        }

        void drawCircle()
        {
            float radius = glControl1.Width > glControl1.Height ? 1f : 1f * glControl1.Width / glControl1.Height;
            Matrix4 Resize = comboTrackballType.SelectedIndex == 0 ? Matrix4.Identity : glControl1.Width > glControl1.Height ? Matrix4.Scale(1.0f * glControl1.Width / glControl1.Height, 1, 1) : Matrix4.Scale(1, 1f * glControl1.Height / glControl1.Width, 1);
            GL.MatrixMode(MatrixMode.Modelview);
            Matrix4 modelview = //Matrix4.CreateTranslation(-center) *
                              Resize *
                              Matrix4.CreateTranslation(0.0f, 0.0f, -1.8f);
            GL.LoadMatrix(ref modelview);
            GL.Color3(System.Drawing.Color.LimeGreen);
            GL.Begin(BeginMode.LineLoop);

            for (int i = 0; i < 360; i++)
            {
                double degInRad = i * Math.PI / 180;
                GL.Vertex2(Math.Cos(degInRad) * radius, Math.Sin(degInRad) * radius);
            }
            GL.End();
        }
    }
}
