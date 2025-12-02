import Axios, { AxiosError, type AxiosRequestConfig } from "axios";
import qs from "qs";

export const AXIOS_INSTANCE = Axios.create({
  baseURL: "http://127.0.0.1:3000",
  paramsSerializer: {
    // Important to use qs instead of the default URLSearchParams
    serialize: (params) => {
      return qs.stringify(params, { arrayFormat: "comma" });
    },
  },
});

export const customInstance = <T>(
  config: AxiosRequestConfig,
  options?: AxiosRequestConfig
): Promise<T> => {
  const source = Axios.CancelToken.source();

  const promise = AXIOS_INSTANCE({
    ...config,
    ...options,
    cancelToken: source.token,
  }).then(({ data }) => data);

  //@ts-expect-error "Because of the way Axios is typed, we need to add a cancel method"
  promise.cancel = () => {
    source.cancel("Query was cancelled");
  };

  return promise;
};

export type ErrorType<Error> = AxiosError<Error>;
