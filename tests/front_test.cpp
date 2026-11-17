/* front_test.cpp -- the listener in front of every model (core/server/front.h).
 *
 *   Read     a request's model comes from the query, a form field, or the body's top-level
 *            string, found without parsing the body
 *   Merged   several models' metrics are one exposition, each family declared once
 *   Reads    what the dashboard polls is told apart from requests for work
 *   Jobs     a finished job is answered by its id with no model, until it expires or is deleted
 *
 * No server and no socket: requests and texts in, answers out.
 */
#include "rad_test.h"

#include "server/front.h"

#include <ctime>
#include <string>

using namespace rad;
using namespace rad::server;

TEST(the_top_level_model_string_is_found_without_a_parse) {
    size_t b = 0, e = 0;
    const std::string body =
        R"({"messages":[{"role":"user","content":"say \"model\": no"}],"n":2,"x":{"model":"inner"},)"
        R"( "model" : "qwen:sql", "stream": true})";
    CHECK(json_top_string(body, "model", &b, &e));
    CHECK_EQ(body.substr(b, e - b), "qwen:sql");
    CHECK(!json_top_string(R"({"messages":[]})", "model", &b, &e));
    CHECK(!json_top_string(R"({"model": 3})", "model", &b, &e));
    CHECK(!json_top_string(R"([{"model":"x"}])", "model", &b, &e));
    CHECK(!json_top_string(R"({"model":"unterminated)", "model", &b, &e));
}

TEST(a_request_names_its_model_by_query_form_or_body) {
    HttpRequest q;
    q.body = R"({"model":"a"})";
    CHECK_EQ(Front::requested_model(q), "a");
    q.query["model"] = "b";
    CHECK_EQ(Front::requested_model(q), "b");
    HttpRequest f;
    f.form.push_back(FormPart{ "model", "", "", "whisper" });
    CHECK_EQ(Front::requested_model(f), "whisper");
    HttpRequest esc;
    esc.body = R"({"model":"a\u002db"})";
    CHECK_EQ(Front::requested_model(esc), "a-b");
    CHECK_EQ(Front::requested_model(HttpRequest{}), "");
}

TEST(the_dashboards_reads_do_not_count_as_wanting_a_model) {
    HttpRequest q;
    for (const char* path : { "/stats", "/sessions", "/graph", "/server_info", "/admin/control-vector" }) {
        q.path = path;
        q.method = "GET";
        CHECK(Front::dashboard_read(q));
        /* The router dispatches a doubled or trailing slash to the same route; so must the
         * read's exemption, or the read wakes a model for a slash. */
        q.path = std::string("/") + path;
        CHECK(Front::dashboard_read(q));
        q.path = std::string(path) + "/";
        CHECK(Front::dashboard_read(q));
    }
    q.path = "/admin/control-vector";
    q.method = "POST";
    CHECK(!Front::dashboard_read(q));
    q.path = "/v1/chat/completions";
    CHECK(!Front::dashboard_read(q));
    q.method = "GET";
    q.path = "/v1/videos/abc";
    CHECK(!Front::dashboard_read(q));
}

/* The set that must not wake a model for a caller its handler will refuse: exactly the routes
 * Server::register_routes guards with local_operator keyless, and nothing beside them. */
TEST(guarded_routes_are_the_local_operator_ones) {
    /* guarded_route now reads the front's own copied routes, so the decision is made by an
     * instance's matcher -- a Front with no models installed still carries the full route table
     * (its prototype does), which is exactly the set whose handlers guard. */
    Front f(TransportOptions{}, 0);
    HttpRequest q;
    q.method = "GET";    q.path = "/diagnostics";          CHECK(f.guarded_route(q));
    /* The router collapses "//x" and "x/" into "/x" (split_path, http.cpp) -- the guard sees
     * the same path the handler reaches, or a slash buys the wake the guard denies. */
    q.path = "//diagnostics";                              CHECK(f.guarded_route(q));
    q.path = "/diagnostics/";                              CHECK(f.guarded_route(q));
    q.method = "POST";   q.path = "/reset_prefix_cache";   CHECK(f.guarded_route(q));
    q.path = "/admin/control-vector";                      CHECK(f.guarded_route(q));
    q.method = "GET";    q.path = "/admin/control-vector"; CHECK(f.guarded_route(q));
    q.method = "DELETE"; q.path = "/sessions";             CHECK(f.guarded_route(q));
    /* The :label form is registered and guarded next to it (server.cpp); the front must not
     * wake a model for a stranger's DELETE /sessions/whatever either. */
    q.path = "/sessions/abc";                              CHECK(f.guarded_route(q));
    q.method = "POST";   q.path = "/v1/load_lora_adapter"; CHECK(f.guarded_route(q));
    q.path = "/v1/unload_lora_adapter";                    CHECK(f.guarded_route(q));
    /* The rest is unguarded: the open dashboard reads stay as they are, and inference keeps
     * waking its model for whoever asks -- that is what a keyless box is for. */
    q.method = "GET";    q.path = "/sessions";             CHECK(!f.guarded_route(q));
    q.path = "/stats";                                     CHECK(!f.guarded_route(q));
    q.method = "POST";   q.path = "/v1/chat/completions";  CHECK(!f.guarded_route(q));
    q.method = "GET";    q.path = "/v1/models";            CHECK(!f.guarded_route(q));
    /* Shapes the :label pattern cannot match are no route at all, so they wake nothing -- but a
     * trailing slash is not such a shape: the router collapses it to the guarded bare route. */
    q.method = "DELETE"; q.path = "/sessions/";            CHECK(f.guarded_route(q));
    q.path = "/sessions/a/b";                              CHECK(!f.guarded_route(q));
}

TEST(a_finished_job_is_answered_by_its_id_with_no_model) {
    Front f(TransportOptions{}, 0);
    const int64_t later = (int64_t)std::time(nullptr) + 100;
    int asked = 0;
    const Handler answer = [&](const HttpRequest&) { ++asked; return HttpResponse::json_body(200, "{}"); };
    f.keep_job("video_a", later, 8, answer);
    f.keep_job("shape_b", later, 8, answer);
    HttpRequest q;
    q.method = "GET";
    q.path = "/v1/videos/video_a";
    CHECK(f.kept_job(q) != nullptr);
    q.path = "/v1/videos/video_a/content";
    CHECK(f.kept_job(q) != nullptr);
    q.path = "/status/shape_b";
    CHECK(f.kept_job(q) != nullptr);
    /* Another job, a prefix of the id, and a route that names no job. */
    q.path = "/v1/videos/video_b";
    CHECK(f.kept_job(q) == nullptr);
    q.path = "/v1/videos/video_";
    CHECK(f.kept_job(q) == nullptr);
    q.path = "/v1/chat/completions";
    CHECK(f.kept_job(q) == nullptr);
    /* A delete is answered once and forgets it. */
    q.method = "DELETE";
    q.path = "/v1/videos/video_a";
    const Handler del = f.kept_job(q);
    CHECK(del != nullptr);
    CHECK_EQ(del(q).status, 200);
    CHECK_EQ(asked, 1);
    q.method = "GET";
    CHECK(f.kept_job(q) == nullptr);
}

TEST(kept_jobs_expire_and_the_oldest_go_past_the_count) {
    Front f(TransportOptions{}, 0);
    const int64_t now = (int64_t)std::time(nullptr);
    const Handler answer = [](const HttpRequest&) { return HttpResponse::json_body(200, "{}"); };
    HttpRequest q;
    q.method = "GET";
    f.keep_job("old", now - 1, 8, answer);
    q.path = "/v1/videos/old";
    CHECK(f.kept_job(q) == nullptr);
    f.keep_job("a", now + 100, 2, answer);
    f.keep_job("b", now + 100, 2, answer);
    f.keep_job("c", now + 100, 2, answer);
    q.path = "/v1/videos/a";
    CHECK(f.kept_job(q) == nullptr);
    q.path = "/v1/videos/b";
    CHECK(f.kept_job(q) != nullptr);
    q.path = "/v1/videos/c";
    CHECK(f.kept_job(q) != nullptr);
}

TEST(metrics_from_several_models_declare_each_family_once) {
    const std::string a =
        "# HELP rad_requests_total Requests.\n# TYPE rad_requests_total counter\n"
        "rad_requests_total{model_name=\"a\"} 3\n"
        "# HELP rad_step_seconds Step time.\n# TYPE rad_step_seconds histogram\n"
        "rad_step_seconds_bucket{model_name=\"a\",le=\"0.1\"} 1\nrad_step_seconds_sum{model_name=\"a\"} 0.05\n";
    const std::string b =
        "# HELP rad_requests_total Requests.\n# TYPE rad_requests_total counter\n"
        "rad_requests_total{model_name=\"b\"} 5\n";
    const std::string m = merge_metrics({ a, b });
    CHECK_EQ(m,
             "# HELP rad_requests_total Requests.\n# TYPE rad_requests_total counter\n"
             "rad_requests_total{model_name=\"a\"} 3\nrad_requests_total{model_name=\"b\"} 5\n"
             "# HELP rad_step_seconds Step time.\n# TYPE rad_step_seconds histogram\n"
             "rad_step_seconds_bucket{model_name=\"a\",le=\"0.1\"} 1\n"
             "rad_step_seconds_sum{model_name=\"a\"} 0.05\n");
}

RAD_TEST_MAIN()
