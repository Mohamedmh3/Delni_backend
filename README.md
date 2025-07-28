# DELNI Backend Project

This is the backend for the DELNI public transportation application. It provides APIs for route finding, real-time updates, and other core features for the DELNI frontend and mobile apps.

## Features
- Advanced route planning and optimization
- Real-time bus and route information
- RESTful API built with Django and Django REST Framework
- MongoDB integration for scalable data storage
- Geospatial queries and location-based services
- CORS support for frontend integration

## Technology Stack
- **Python 3.11+**
- **Django 5.2.4**
- **Django REST Framework**
- **MongoDB** (via `pymongo`)
- **Geopy** for geospatial calculations
- **Gunicorn** for production WSGI serving

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd backend_bus_pr
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Create a `.env` file or set environment variables as needed (see `python-decouple` usage).
   - Set up your MongoDB connection string and any other required settings.

4. **Run migrations (if using Django models):**
   ```sh
   python manage.py migrate
   ```

5. **Start the development server:**
   ```sh
   python manage.py runserver
   ```

## Deployment (Railway)
- Configure your Railway project to use Python and install dependencies from `requirements.txt`.
- Set environment variables in Railway dashboard (e.g., `MONGO_URI`, `DJANGO_SECRET_KEY`).
- Use Gunicorn for production serving:
  ```sh
  gunicorn backend_bus_pr.wsgi:application --bind 0.0.0.0:$PORT
  ```
- Make sure to connect your Railway project to your MongoDB instance.

## API Endpoints
- Route finding: `/api/graph-route/`
- Real-time info: `/api/real-time-info/`
- Additional endpoints as defined in `urls.py`

## Contributing
Pull requests and issues are welcome! Please follow best practices and ensure all code is tested.

---

**Built for the DELNI community.** 