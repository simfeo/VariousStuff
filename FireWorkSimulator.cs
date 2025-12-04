// Online C# Editor for free
// Write, Edit and Run your C# code using C# Online Compiler

using System;

public class FireWorkSimulator
{

	public static class RandomUtils
	{
		private static readonly Random _random = new Random();

		public static float RandomRange(float min, float max)
		{
			if (min > max)
				throw new ArgumentException("min must be less than or equal to max");

			double range = max - min;
			return (float)(min + _random.NextDouble() * range);
		}
		
		public static int RandomRange(int min, int max)
		{
			if (min > max)
				throw new ArgumentException("min must be less than or equal to max");

			return _random.Next(min, max);
		}
	}
	
    public class Vector2 : IEquatable<Vector2>
    {
        /// <summary>
        /// The x coordinate of the vector.
        /// </summary>
        public float x;

        /// <summary>
        /// The y coordinate of the vector.
        /// </summary>
        public float y;

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector2"/> struct.
        /// </summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        public Vector2(float x, float y)
        {
            this.x = x;
            this.y = y;
        }

        /// <summary>
        /// Gets the zero vector (0,0).
        /// </summary>
        public static Vector2 Zero => new Vector2(0f, 0f);

        /// <summary>
        /// Gets the vector (1,1).
        /// </summary>
        public static Vector2 One => new Vector2(1f, 1f);

        /// <summary>
        /// Gets the unit vector pointing right (1,0).
        /// </summary>
        public static Vector2 UnitX => new Vector2(1f, 0f);

        /// <summary>
        /// Gets the unit vector pointing up (0,1).
        /// </summary>
        public static Vector2 UnitY => new Vector2(0f, 1f);

        /// <summary>
        /// Gets the squared length of the vector.
        /// </summary>
        public float LengthSquared => x * x + y * y;

        /// <summary>
        /// Gets the length (magnitude) of the vector.
        /// </summary>
        public float Length => MathF.Sqrt(LengthSquared);

        /// <summary>
        /// Returns a normalized (unit length) version of this vector.
        /// If the vector is zero, returns the zero vector.
        /// </summary>
        public Vector2 normalized
        {
            get
            {
                float length = Length;
                if (length == 0f)
                    return Zero;
                return this / length;
            }
        }
		
		public float magnitude
        {
            get
            {
                return Length;
            }
        }
			
		public void Normalize()
        {
			float length = Length;
			if (length == 0f)
				return;
			Vector2 tmp = this / length;
            x = tmp.x;
			y = tmp.y;
        }

        /// <summary>
        /// Returns the dot product of two vectors.
        /// </summary>
        public static float Dot(Vector2 a, Vector2 b) => a.x * b.x + a.y * b.y;

        /// <summary>
        /// Returns the distance between two vectors.
        /// </summary>
        public static float Distance(Vector2 a, Vector2 b) => (a - b).Length;

        /// <summary>
        /// Returns the angle in radians between two vectors.
        /// </summary>
        public static float AngleBetween(Vector2 a, Vector2 b)
        {
            float dot = Dot(a.normalized, b.normalized);
            dot = Clamp(dot, -1f, 1f);
            return MathF.Acos(dot);
        }

        /// <summary>
        /// Linearly interpolates between two vectors by t (0 to 1).
        /// </summary>
        public static Vector2 Lerp(Vector2 a, Vector2 b, float t)
        {
            t = Clamp(t, 0f, 1f);
            return new Vector2(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t
            );
        }
		
		static float Clamp(float in_f, float min, float max)
		{
			if (in_f > max)
				return max;
			if (in_f < min)
				return min;
			return in_f;
		}

        /// <summary>
        /// Creates a new vector with optionally replaced components.
        /// </summary>
        public Vector2 With(float? x = null, float? y = null)
        {
            return new Vector2(x ?? this.x, y ?? this.y);
        }

        /// <summary>
        /// Deconstructs the vector into its components.
        /// </summary>
        public void Deconstruct(out float x, out float y)
        {
            x = this.x;
            y = this.y;
        }

        // Operator overloads

        public static Vector2 operator +(Vector2 a, Vector2 b) => new Vector2(a.x + b.x, a.y + b.y);

        public static Vector2 operator -(Vector2 a, Vector2 b) => new Vector2(a.x - b.x, a.y - b.y);

        public static Vector2 operator -(Vector2 v) => new Vector2(-v.x, -v.y);

        public static Vector2 operator *(Vector2 v, float scalar) => new Vector2(v.x * scalar, v.y * scalar);

        public static Vector2 operator *(float scalar, Vector2 v) => v * scalar;

        public static Vector2 operator /(Vector2 v, float scalar) => new Vector2(v.x / scalar, v.y / scalar);

        // Equality members

        public bool Equals(Vector2 other) => x.Equals(other.x) && y.Equals(other.y);

        public override bool Equals(object? obj) => obj is Vector2 other && Equals(other);

        public override int GetHashCode() => HashCode.Combine(x, y);

        public static bool operator ==(Vector2 left, Vector2 right) => left.Equals(right);

        public static bool operator !=(Vector2 left, Vector2 right) => !(left == right);

        /// <summary>
        /// Returns a string representation of the vector.
        /// </summary>
        public override string ToString() => $"({x}, {y})";
    }
	
	public class FireWork
	{
		public static Vector2 gravity = new Vector2(0.0f, 1.0f);
	}
	
    
class FireWorkParticle
{
    public Vector2 coordinate;
    public Vector2 direction;
    int current_tik = 0;
    public float acceleration;
    public float velocity = 0.0f;
    public int accelerationMaxTime;
    public int lifeMaxTime;

    public FireWorkParticle()
    {
        // color = availableColors[Random.Range(0, availableColors.Length -1)];
        coordinate = new Vector2(RandomUtils.RandomRange(64, 192), 0);
        direction = new Vector2(RandomUtils.RandomRange(-0.1f, 0.1f), 1.0f);
        direction.Normalize();
        acceleration = RandomUtils.RandomRange(2.0f, 5.0f);
        accelerationMaxTime = RandomUtils.RandomRange(20, 30);
        lifeMaxTime = RandomUtils.RandomRange(1, 4) + accelerationMaxTime;
    }

    public bool isAlive()
    {
        return lifeMaxTime > current_tik;
    }


    public void Update()
    {
        if (lifeMaxTime < current_tik)
        {
            //not alive
            return ;
        }

        if (current_tik >= accelerationMaxTime)
        {
            acceleration = 0;
        }

        Vector2 result = coordinate;
        System.Console.WriteLine(coordinate);
        Vector2 velocityVec = direction * velocity + direction * acceleration + FireWork.gravity;
        coordinate = coordinate + velocityVec;
        direction = velocityVec.normalized;
        velocity = velocityVec.magnitude;
        ++current_tik;
    }
}
	public static void Main(string[] args)
    {
        FireWorkParticle particle = new();
		while ( particle.isAlive() )
		{
			particle.Update();
		}
		System.Console.WriteLine("Hello, World!");
    }
}
